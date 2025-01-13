import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from thop import profile
import random

class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_layers=12,
        top_k=1,
        prune_ratio=0.0,
        target_prune_ratio=0.5,
        prune_step=0.05,
        score_method="attention",
    ):
        super(VisionTransformer, self).__init__()

        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True)

        self.backbone.patch_embed.proj = nn.Conv2d(1, 384, kernel_size=16, stride=16)

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.num_layers = num_layers
        self.transformer_layers = nn.ModuleList(
            [self.backbone.blocks[i] for i in range(self.num_layers)]
        )
        
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        
        self.top_k = top_k
        self.score_method = score_method
        self.prune_ratio = prune_ratio
        self.prune_step = prune_step
        self.target_prune_ratio = target_prune_ratio

        self.attention_scores = None
        self.backbone.blocks = None
        self.register_hooks()

    def hook_fn(self, module, input, output):
        self.attention_scores = module.attn_drop(output)
    # def hook_fn(self, module, input, output):
    #     self.attention_scores = output[1]
    #     print("Attention scores shape:", self.attention_scores.shape)

    def register_hooks(self):
        for block in self.transformer_layers:
            block.attn.register_forward_hook(self.hook_fn)

    def prune_heads(self):
        current_prune_ratio = min(self.prune_ratio, self.target_prune_ratio)
        for block in self.transformer_layers:
            attn_weights = self.attention_scores 
            importance_scores = attn_weights.mean(dim=1).mean(dim=1).cpu().numpy()
            num_heads_to_prune = int(block.attn.num_heads * current_prune_ratio)
            pruned_heads = importance_scores.argsort()[:num_heads_to_prune]
            for head in pruned_heads:
                block.attn.head_mask[head] = 0
        
        self.prune_ratio += self.prune_step

    def random_prune_heads(self):
        for block in self.transformer_layers:
            num_heads = block.attn.num_heads
            num_heads_to_prune = num_heads // 3 
            pruned_heads = random.sample(range(num_heads), num_heads_to_prune)

            for head in pruned_heads:
                self.attention_weights[:, head, :, :] = 0.0

    def forward(self, x):
        # self.register_hooks()

        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed.shape[1] == 197 and x.shape[1] == 196:
            pos_embed = self.backbone.pos_embed[:, 1:, :] 
        else:
            pos_embed = self.backbone.pos_embed
    
        x = self.backbone.pos_drop(x + pos_embed)

        for i, block in enumerate(self.transformer_layers):
            # print(block)
            x = block(x)
            if i%2 ==1:
                if self.score_method == "attention":
                    attn_scores = self.attention_scores.mean(dim=-1)
                    topk_indices = attn_scores.topk(
                        int(self.top_k * attn_scores.size(1)), dim=1, largest=True
                    ).indices
                    if topk_indices.max() >= x.size(1):
                        raise ValueError("topk_indices contains out of bounds index")
    
                    bs = x.size(0)
                    batch_indices = (
                        torch.arange(bs)
                        .unsqueeze(-1)
                        .expand(-1, topk_indices.size(1))
                        .to(x.device)
                    )
    
                    informative_tokens = x[batch_indices, topk_indices]
    
                    non_informative_indices = torch.ones_like(attn_scores, dtype=bool)
                    non_informative_indices[batch_indices, topk_indices] = False
                    non_informative_tokens = x[non_informative_indices].view(
                        bs, -1, x.size(-1)
                    )
                    x = informative_tokens
                    # if non_informative_tokens.size(1) > 0:
                    #     non_informative_scores = attn_scores[non_informative_indices].view(
                    #         bs, -1
                    #     )
                    #     weighted_sum = (
                    #         non_informative_tokens * non_informative_scores.unsqueeze(-1)
                    #     ).sum(dim=1)
                    #     sum_scores = non_informative_scores.sum(dim=1).unsqueeze(-1)
                    #     # sum_scores = non_informative_scores.sum(dim=1).unsqueeze(-1)
                    #     sum_scores = torch.clamp(sum_scores, min=1e-5)  # Clamping to avoid zero values

                    #     package_token = weighted_sum / (sum_scores+1e-5)
                    #     x = torch.cat(
                    #         [informative_tokens, package_token.unsqueeze(1)], dim=1
                    #     )
                    # else:
                    #     x = informative_tokens

        features = x.mean(dim=1)
        gaze_dir = F.relu(self.fc1(features))
        gaze_dir = F.relu(self.fc2(gaze_dir))
        gaze_dir = F.relu(self.fc3(gaze_dir))
        gaze_dir = self.fc4(gaze_dir)

        return gaze_dir
    
    def forward_timer(self, x, starters=None, enders=None):
        # self.register_hooks()

        if starters is None or enders is None:
            raise ValueError("starters and enders must be provided for timing")

        starters[0].record()  # Start timing for patch embedding
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed.shape[1] == 197 and x.shape[1] == 196:
            pos_embed = self.backbone.pos_embed[:, 1:, :] 
        else:
            pos_embed = self.backbone.pos_embed
        x = self.backbone.pos_drop(x + pos_embed)
        enders[0].record()  # End timing for patch embedding

        for i, block in enumerate(self.transformer_layers):
            starters[i+1].record()  # Start timing for this transformer block
            x = block(x)
            if i % 2 == 1 and self.score_method == "attention":
                attn_scores = self.attention_scores.mean(dim=-1)
                topk_indices = attn_scores.topk(
                    int(self.top_k * attn_scores.size(1)), dim=1, largest=True
                ).indices
                if topk_indices.max() >= x.size(1):
                    raise ValueError("topk_indices contains out of bounds index")

                bs = x.size(0)
                batch_indices = (
                    torch.arange(bs)
                    .unsqueeze(-1)
                    .expand(-1, topk_indices.size(1))
                    .to(x.device)
                )

                informative_tokens = x[batch_indices, topk_indices]

                non_informative_indices = torch.ones_like(attn_scores, dtype=bool)
                non_informative_indices[batch_indices, topk_indices] = False
                non_informative_tokens = x[non_informative_indices].view(
                    bs, -1, x.size(-1)
                )
                x = informative_tokens
                
                features = x.mean(dim=1)
                gaze_dir = F.relu(self.fc1(features))
                gaze_dir = F.relu(self.fc2(gaze_dir))
                gaze_dir = F.relu(self.fc3(gaze_dir))
                gaze_dir = self.fc4(gaze_dir)
            enders[i+1].record()  # End timing for this transformer block

        starters[len(self.transformer_layers)+1].record()  # Start timing for final layers
        features = x.mean(dim=1)
        gaze_dir = F.relu(self.fc1(features))
        gaze_dir = F.relu(self.fc2(gaze_dir))
        gaze_dir = F.relu(self.fc3(gaze_dir))
        gaze_dir = self.fc4(gaze_dir)
        enders[len(self.transformer_layers)+1].record()  # End timing for final layers

        return gaze_dir
        

class VisionTransformerFoveated(nn.Module):
    def __init__(
        self,
        num_layers=12,
        top_k=1,
        prune_ratio=0.0,
        target_prune_ratio=0.5,
        prune_step=0.05,
        score_method="attention",
    ):
        super(VisionTransformerFoveated, self).__init__()

        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True)

        self.backbone.patch_embed.proj = nn.Conv2d(1, 384, kernel_size=16, stride=16)

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.num_layers = num_layers
        self.transformer_layers = nn.ModuleList(
            [self.backbone.blocks[i] for i in range(self.num_layers)]
        )
        
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        
        self.top_k = top_k
        self.score_method = score_method
        self.prune_ratio = prune_ratio
        self.prune_step = prune_step
        self.target_prune_ratio = target_prune_ratio

        self.attention_scores = None
        self.backbone.blocks = None
        self.register_hooks()

    def hook_fn(self, module, input, output):
        self.attention_scores = module.attn_drop(output)
    # def hook_fn(self, module, input, output):
    #     self.attention_scores = output[1]
    #     print("Attention scores shape:", self.attention_scores.shape)

    def register_hooks(self):
        for block in self.transformer_layers:
            block.attn.register_forward_hook(self.hook_fn)

    def prune_heads(self):
        current_prune_ratio = min(self.prune_ratio, self.target_prune_ratio)
        for block in self.transformer_layers:
            attn_weights = self.attention_scores 
            importance_scores = attn_weights.mean(dim=1).mean(dim=1).cpu().numpy()
            num_heads_to_prune = int(block.attn.num_heads * current_prune_ratio)
            pruned_heads = importance_scores.argsort()[:num_heads_to_prune]
            for head in pruned_heads:
                block.attn.head_mask[head] = 0
        
        self.prune_ratio += self.prune_step 

    def random_prune_heads(self):
        for block in self.transformer_layers:
            num_heads = block.attn.num_heads
            num_heads_to_prune = num_heads // 3 
            pruned_heads = random.sample(range(num_heads), num_heads_to_prune)

            for head in pruned_heads:
                self.attention_weights[:, head, :, :] = 0.0

    def forward(self, x):
        # self.register_hooks()

        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed.shape[1] == 197 and x.shape[1] == 196:
            pos_embed = self.backbone.pos_embed[:, 1:, :] 
        else:
            pos_embed = self.backbone.pos_embed
    
        x = self.backbone.pos_drop(x + pos_embed)
        # create a tensor of shape (transformer_layers//2 , 2)
        # outputs = torch.zeros((self.num_layers//2, 2))
        outputs = []
        for i, block in enumerate(self.transformer_layers):
            # print(block)
            x = block(x)
            if i%2 ==1:
                if self.score_method == "attention":
                    attn_scores = self.attention_scores.mean(dim=-1)
                    topk_indices = attn_scores.topk(
                        int(self.top_k * attn_scores.size(1)), dim=1, largest=True
                    ).indices
                    if topk_indices.max() >= x.size(1):
                        raise ValueError("topk_indices contains out of bounds index")
    
                    bs = x.size(0)
                    batch_indices = (
                        torch.arange(bs)
                        .unsqueeze(-1)
                        .expand(-1, topk_indices.size(1))
                        .to(x.device)
                    )
    
                    informative_tokens = x[batch_indices, topk_indices]
    
                    non_informative_indices = torch.ones_like(attn_scores, dtype=bool)
                    non_informative_indices[batch_indices, topk_indices] = False
                    non_informative_tokens = x[non_informative_indices].view(
                        bs, -1, x.size(-1)
                    )
                    x = informative_tokens
            features = x.mean(dim=1)
            gaze_dir = F.relu(self.fc1(features))
            gaze_dir = F.relu(self.fc2(gaze_dir))
            gaze_dir = F.relu(self.fc3(gaze_dir))
            gaze_dir = self.fc4(gaze_dir)
            # append a deep copy of the gaze_dir tensor to python list outputs
            # outputs[i//2,:] = gaze_dir.clone()
            outputs.append(gaze_dir.clone())

                    # if non_informative_tokens.size(1) > 0:
                    #     non_informative_scores = attn_scores[non_informative_indices].view(
                    #         bs, -1
                    #     )
                    #     weighted_sum = (
                    #         non_informative_tokens * non_informative_scores.unsqueeze(-1)
                    #     ).sum(dim=1)
                    #     sum_scores = non_informative_scores.sum(dim=1).unsqueeze(-1)
                    #     # sum_scores = non_informative_scores.sum(dim=1).unsqueeze(-1)
                    #     sum_scores = torch.clamp(sum_scores, min=1e-5)  # Clamping to avoid zero values

                    #     package_token = weighted_sum / (sum_scores+1e-5)
                    #     x = torch.cat(
                    #         [informative_tokens, package_token.unsqueeze(1)], dim=1
                    #     )
                    # else:
                    #     x = informative_tokens

        # features = x.mean(dim=1)
        # gaze_dir = F.relu(self.fc1(features))
        # gaze_dir = F.relu(self.fc2(gaze_dir))
        # gaze_dir = F.relu(self.fc3(gaze_dir))
        # gaze_dir = self.fc4(gaze_dir)
        
        # make the python list outputs a tensor
        outputs = torch.stack(outputs)
        return outputs
    
    def forward_timer(self, x, starters=None, enders=None):
        # self.register_hooks()

        if starters is None or enders is None:
            raise ValueError("starters and enders must be provided for timing")

        starters[0].record()  # Start timing for patch embedding
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed.shape[1] == 197 and x.shape[1] == 196:
            pos_embed = self.backbone.pos_embed[:, 1:, :] 
        else:
            pos_embed = self.backbone.pos_embed
        x = self.backbone.pos_drop(x + pos_embed)
        enders[0].record()  # End timing for patch embedding

        for i, block in enumerate(self.transformer_layers):
            starters[i+1].record()  # Start timing for this transformer block
            x = block(x)
            if i % 2 == 1 and self.score_method == "attention":
                attn_scores = self.attention_scores.mean(dim=-1)
                topk_indices = attn_scores.topk(
                    int(self.top_k * attn_scores.size(1)), dim=1, largest=True
                ).indices
                if topk_indices.max() >= x.size(1):
                    raise ValueError("topk_indices contains out of bounds index")

                bs = x.size(0)
                batch_indices = (
                    torch.arange(bs)
                    .unsqueeze(-1)
                    .expand(-1, topk_indices.size(1))
                    .to(x.device)
                )

                informative_tokens = x[batch_indices, topk_indices]

                non_informative_indices = torch.ones_like(attn_scores, dtype=bool)
                non_informative_indices[batch_indices, topk_indices] = False
                non_informative_tokens = x[non_informative_indices].view(
                    bs, -1, x.size(-1)
                )
                x = informative_tokens
                
                features = x.mean(dim=1)
                gaze_dir = F.relu(self.fc1(features))
                gaze_dir = F.relu(self.fc2(gaze_dir))
                gaze_dir = F.relu(self.fc3(gaze_dir))
                gaze_dir = self.fc4(gaze_dir)
            enders[i+1].record()  # End timing for this transformer block

        starters[len(self.transformer_layers)+1].record()  # Start timing for final layers
        features = x.mean(dim=1)
        gaze_dir = F.relu(self.fc1(features))
        gaze_dir = F.relu(self.fc2(gaze_dir))
        gaze_dir = F.relu(self.fc3(gaze_dir))
        gaze_dir = self.fc4(gaze_dir)
        enders[len(self.transformer_layers)+1].record()  # End timing for final layers

        return gaze_dir

class ResNetTracking(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        in_channels=1,
        num_layers=4,  # Number of ResNet stages to use
        top_k=1,
        prune_ratio=0.0,
        target_prune_ratio=0.5,
        prune_step=0.05,
        score_method="feature_map",
    ):
        super(ResNetTracking, self).__init__()

        # Initialize ResNet backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=in_channels)

        # Select layers from ResNet for feature extraction
        # Example for resnet50: layers are layer1, layer2, layer3, layer4
        self.layer_names = ["layer1", "layer2", "layer3", "layer4"][:num_layers]
        self.layers = nn.ModuleList([getattr(self.backbone, layer) for layer in self.layer_names])

        # Define fully connected layers based on the backbone's output dimensions
        backbone_output_dim = self.get_backbone_output_dim(backbone_name)
        self.fc1 = nn.Linear(backbone_output_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

        self.top_k = top_k
        self.score_method = score_method
        self.prune_ratio = prune_ratio
        self.prune_step = prune_step
        self.target_prune_ratio = target_prune_ratio

        # Placeholder for attention or feature map scores if needed
        self.feature_scores = None

        # Register hooks if necessary (adapt based on your foveation method)
        # For ResNet, this might involve registering hooks on specific layers to capture feature maps
        self.register_hooks()

    def get_backbone_output_dim(self, backbone_name):
        # Define output dimensions based on backbone
        if backbone_name.startswith("resnet50"):
            return 2048
        elif backbone_name.startswith("resnet34"):
            return 512
        # Add more mappings if using different ResNet variants
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def hook_fn(self, module, input, output):
        # Example: Capture feature maps for foveation
        self.feature_scores = output.mean(dim=(2, 3))  # Global average pooling as an example

    def register_hooks(self):
        for layer in self.layers:
            layer.register_forward_hook(self.hook_fn)

    def prune_features(self):
        # Implement feature pruning based on self.feature_scores
        current_prune_ratio = min(self.prune_ratio, self.target_prune_ratio)
        num_features_to_prune = int(self.feature_scores.size(1) * current_prune_ratio)
        pruned_features = self.feature_scores.argsort()[:, :num_features_to_prune]

        # Example pruning: Zero out the least important features
        mask = torch.ones_like(self.feature_scores)
        mask[:, pruned_features] = 0
        # Apply mask to features (this is a simplified example)
        # You might need to adapt this based on where and how you want to apply pruning

        self.prune_ratio += self.prune_step

    def forward(self, x):
        features = []
                # Pass through initial ResNet layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        # Example: Use the last layer's output
        x = features[-1]
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

        gaze_dir = F.relu(self.fc1(x))
        gaze_dir = F.relu(self.fc2(gaze_dir))
        gaze_dir = F.relu(self.fc3(gaze_dir))
        gaze_dir = self.fc4(gaze_dir)

        return gaze_dir

    def forward_timer(self, x, starters=None, enders=None):
        if starters is None or enders is None:
            raise ValueError("starters and enders must be provided for timing")

        starters[0].record()  # Start timing for backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        enders[0].record()  # End timing for initial backbone layers

        for i, layer in enumerate(self.layers):
            starters[i+1].record()  # Start timing for this ResNet layer
            x = layer(x)
            enders[i+1].record()  # End timing for this ResNet layer

        starters[len(self.layers)+1].record()  # Start timing for final layers
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        gaze_dir = F.relu(self.fc1(x))
        gaze_dir = F.relu(self.fc2(gaze_dir))
        gaze_dir = F.relu(self.fc3(gaze_dir))
        gaze_dir = self.fc4(gaze_dir)
        enders[len(self.layers)+1].record()  # End timing for final layers

        return gaze_dir
    

class ResNetFoveated(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        in_channels=1,
        num_layers=4,  # Number of ResNet stages to use
        top_k=1,
        prune_ratio=0.0,
        target_prune_ratio=0.5,
        prune_step=0.05,
        score_method="feature_map",
    ):
        super(ResNetFoveated, self).__init__()

        # Initialize ResNet backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=in_channels)

        # Select layers from ResNet for feature extraction
        # Example for resnet50: layers are layer1, layer2, layer3, layer4
        self.layer_names = ["layer1", "layer2", "layer3", "layer4"][:num_layers]
        self.layers = nn.ModuleList([getattr(self.backbone, layer) for layer in self.layer_names])

        # Define fully connected layers based on the backbone's output dimensions
        backbone_output_dim = self.get_backbone_output_dim(backbone_name)
        self.fc1 = nn.Linear(backbone_output_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

        self.fc1_0 = nn.Linear(64, 512)
        self.fc1_1 = nn.Linear(128, 512)
        self.fc1_2 = nn.Linear(256, 512)

        self.top_k = top_k
        self.score_method = score_method
        self.prune_ratio = prune_ratio
        self.prune_step = prune_step
        self.target_prune_ratio = target_prune_ratio

        # Placeholder for attention or feature map scores if needed
        self.feature_scores = None

        # Register hooks if necessary (adapt based on your foveation method)
        # For ResNet, this might involve registering hooks on specific layers to capture feature maps
        self.register_hooks()

    def get_backbone_output_dim(self, backbone_name):
        # Define output dimensions based on backbone
        if backbone_name.startswith("resnet50"):
            return 2048
        elif backbone_name.startswith("resnet34"):
            return 512
        # Add more mappings if using different ResNet variants
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def hook_fn(self, module, input, output):
        # Example: Capture feature maps for foveation
        self.feature_scores = output.mean(dim=(2, 3))  # Global average pooling as an example

    def register_hooks(self):
        for layer in self.layers:
            layer.register_forward_hook(self.hook_fn)

    def prune_features(self):
        # Implement feature pruning based on self.feature_scores
        current_prune_ratio = min(self.prune_ratio, self.target_prune_ratio)
        num_features_to_prune = int(self.feature_scores.size(1) * current_prune_ratio)
        pruned_features = self.feature_scores.argsort()[:, :num_features_to_prune]

        # Example pruning: Zero out the least important features
        mask = torch.ones_like(self.feature_scores)
        mask[:, pruned_features] = 0
        # Apply mask to features (this is a simplified example)
        # You might need to adapt this based on where and how you want to apply pruning

        self.prune_ratio += self.prune_step

    def forward(self, x):
        outputs = []
                # Pass through initial ResNet layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        layeridx = 0
        for layer in self.layers:
            x = layer(x)

            gazedir = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            if layeridx == 0:
                gaze_dir = F.relu(self.fc1_0(gazedir))
            elif layeridx == 1:
                gaze_dir = F.relu(self.fc1_1(gazedir))
            elif layeridx == 2:
                gaze_dir = F.relu(self.fc1_2(gazedir))
            else:
                gaze_dir = F.relu(self.fc1(gazedir))
            gaze_dir = F.relu(self.fc2(gaze_dir))
            gaze_dir = F.relu(self.fc3(gaze_dir))
            gaze_dir = self.fc4(gaze_dir)
            
            outputs.append(gaze_dir.clone())
            layeridx += 1
        
        # make the python list outputs a tensor
        outputs = torch.stack(outputs)
        return outputs

        
        # x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

        # gaze_dir = F.relu(self.fc1(x))
        # gaze_dir = F.relu(self.fc2(gaze_dir))
        # gaze_dir = F.relu(self.fc3(gaze_dir))
        # gaze_dir = self.fc4(gaze_dir)



        # return gaze_dir

    def forward_timer(self, x, starters=None, enders=None):
        if starters is None or enders is None:
            raise ValueError("starters and enders must be provided for timing")

        starters[0].record()  # Start timing for backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        enders[0].record()  # End timing for initial backbone layers

        for i, layer in enumerate(self.layers):
            starters[i+1].record()  # Start timing for this ResNet layer
            x = layer(x)

            gazedir = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            if i == 0:
                gaze_dir = F.relu(self.fc1_0(gazedir))
            elif i == 1:
                gaze_dir = F.relu(self.fc1_1(gazedir))
            elif i == 2:
                gaze_dir = F.relu(self.fc1_2(gazedir))
            else:
                gaze_dir = F.relu(self.fc1(gazedir))
            gaze_dir = F.relu(self.fc2(gaze_dir))
            gaze_dir = F.relu(self.fc3(gaze_dir))
            gaze_dir = self.fc4(gaze_dir)
            
            enders[i+1].record()  # End timing for this ResNet layer

        starters[len(self.layers)+1].record()  # Start timing for final layers
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        gaze_dir = F.relu(self.fc1(x))
        gaze_dir = F.relu(self.fc2(gaze_dir))
        gaze_dir = F.relu(self.fc3(gaze_dir))
        gaze_dir = self.fc4(gaze_dir)
        enders[len(self.layers)+1].record()  # End timing for final layers

        return gaze_dir
    

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     vit = VisionTransformer(score_method="attention", top_k=0.8, num_layers=6).to(
#         device
#     )

#     input_image = torch.randn(1, 1, 224, 224).to(device)
#     output = vit(input_image)
#     print(output.shape)

#     flops, params = profile(vit, inputs=(input_image,))

#     print(f"Total Params: {params}")
#     print(f"Total FLOPs: {flops}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFoveated(
        backbone_name="resnet34",
        pretrained=True,
        in_channels=1,
        num_layers=4,
        top_k=1.0
    ).to(device)
    
    input_image = torch.randn(50, 1, 224, 224).to(device)
    output = model(input_image)
    print(output.shape)  # Expected: [50, 2]