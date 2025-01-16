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



# DeepVOG, translated from keras

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, filter_size, filters_num, layer_num, block_type, stage, stride=1):
        super(EncodingBlock, self).__init__()
        
        self.conv_name_base = f'conv_{block_type}{stage}_'
        self.bn_name_base = f'bn_{block_type}{stage}_'
        
        layers = []
        for i in range(1, layer_num + 1):
            layers.append(nn.Conv2d(in_channels, filters_num, kernel_size=filter_size, stride=stride if i == 1 else 1, padding='same'))
            layers.append(nn.BatchNorm2d(filters_num))
            if i != layer_num:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ReLU(inplace=True))
        
        self.main_path = nn.Sequential(*layers)
        
        # Downsampling layer
        self.downsample = nn.Sequential(
            nn.Conv2d(filters_num, filters_num * 2, kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(filters_num * 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, X):
        X_main = self.main_path(X)
        X_downed = self.downsample(X_main)
        return X_main, X_downed


class DecodingBlock(nn.Module):
    def __init__(self, in_channels, concat_channels, filter_size, filters_num, layer_num, block_type, stage, stride=1, up_sampling=True):
        super(DecodingBlock, self).__init__()
        
        self.conv_name_base = f'conv_{block_type}{stage}_'
        self.bn_name_base = f'bn_{block_type}{stage}_'
        self.up_sampling = up_sampling
        
        # Upsampling path
        if self.up_sampling:
            self.up_conv = nn.ConvTranspose2d(in_channels, filters_num, kernel_size=(2, 2), stride=(2, 2), padding=0)
            self.up_bn = nn.BatchNorm2d(filters_num)
            self.up_relu = nn.ReLU(inplace=True)
        
        # Calculate input channels for main path after concatenation
        main_path_in_channels = filters_num + concat_channels if self.up_sampling else in_channels
        
        # Main path layers
        layers_main = []
        current_channels = main_path_in_channels
        for i in range(1, layer_num + 1):
            layers_main.append(nn.Conv2d(current_channels, filters_num, kernel_size=filter_size, stride=stride, padding='same'))
            layers_main.append(nn.BatchNorm2d(filters_num))
            if i != layer_num:
                layers_main.append(nn.ReLU(inplace=True))
            current_channels = filters_num
            
        layers_main.append(nn.ReLU(inplace=True))
        self.main_path = nn.Sequential(*layers_main)
        
    def forward(self, X, X_jump):
        if self.up_sampling:
            X = self.up_conv(X)
            X = self.up_bn(X)
            X = self.up_relu(X)
            
            if X_jump is not None:
                X = torch.cat([X, X_jump], dim=1)
        
        X = self.main_path(X)
        return X
    
    
class DeepVOG_net(nn.Module):
    def __init__(self, input_shape=(240, 320, 3), filter_size=(3,3)):
        super(DeepVOG_net, self).__init__()
        
        self.input_shape = input_shape
        self.filter_size = filter_size
        
        # Encoding Stream
        self.enc_block1 = EncodingBlock(in_channels=3, filter_size=filter_size, filters_num=16, layer_num=1, block_type="down", stage=1)
        self.enc_block2 = EncodingBlock(in_channels=32, filter_size=filter_size, filters_num=32, layer_num=1, block_type="down", stage=2)
        self.enc_block3 = EncodingBlock(in_channels=64, filter_size=filter_size, filters_num=64, layer_num=1, block_type="down", stage=3)
        self.enc_block4 = EncodingBlock(in_channels=128, filter_size=filter_size, filters_num=128, layer_num=1, block_type="down", stage=4)
        
        # # Decoding Stream
        # self.dec_block1 = DecodingBlock(in_channels=256, concat_channels=0, filter_size=filter_size, filters_num=256, layer_num=1, block_type="up", stage=1, stride=1)
        # self.dec_block2 = DecodingBlock(in_channels=256, concat_channels=128, filter_size=filter_size, filters_num=256, layer_num=1, block_type="up", stage=2, stride=1)
        # self.dec_block3 = DecodingBlock(in_channels=256, concat_channels=64, filter_size=filter_size, filters_num=128, layer_num=1, block_type="up", stage=3, stride=1)
        # self.dec_block4 = DecodingBlock(in_channels=128, concat_channels=32, filter_size=filter_size, filters_num=64, layer_num=1, block_type="up", stage=4, stride=1)
        # self.dec_block5 = DecodingBlock(in_channels=64, concat_channels=16, filter_size=filter_size, filters_num=32, layer_num=1, block_type="up", stage=5, stride=1, up_sampling=False)
        
        # Decoding Stream
        self.dec_block1 = DecodingBlock(in_channels=256, concat_channels=128, filter_size=filter_size, filters_num=256, layer_num=1, block_type="up", stage=1, stride=1)
        self.dec_block2 = DecodingBlock(in_channels=256, concat_channels=64, filter_size=filter_size, filters_num=256, layer_num=1, block_type="up", stage=2, stride=1)
        self.dec_block3 = DecodingBlock(in_channels=256, concat_channels=32, filter_size=filter_size, filters_num=128, layer_num=1, block_type="up", stage=3, stride=1)
        self.dec_block4 = DecodingBlock(in_channels=128, concat_channels=16, filter_size=filter_size, filters_num=64, layer_num=1, block_type="up", stage=4, stride=1)
        self.dec_block5 = DecodingBlock(in_channels=64, concat_channels=0, filter_size=filter_size, filters_num=32, layer_num=1, block_type="up", stage=5, stride=1, up_sampling=False)
        
        # Output layer
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1,1), stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, X):
        # Encoding Stream
        X_jump1, X_out = self.enc_block1(X)
        X_jump2, X_out = self.enc_block2(X_out)
        X_jump3, X_out = self.enc_block3(X_out)
        X_jump4, X_out = self.enc_block4(X_out)
        
        # Decoding Stream
        X_out = self.dec_block1(X_out, None)      # No skip connection for the first decoding block
        X_out = self.dec_block2(X_out, X_jump4)
        X_out = self.dec_block3(X_out, X_jump3)
        X_out = self.dec_block4(X_out, X_jump2)
        X_out = self.dec_block5(X_out, X_jump1)
        
        # Output layer
        X_out = self.conv_out(X_out)
        X_out = self.softmax(X_out)
        
        return X_out



class DeepVOGFoveated(nn.Module):
    def __init__(
        self,
        backbone_name="DeepVOG_net",
        pretrained=False,  # DeepVOG_net may not have pretrained weights
        in_channels=3,
        in_height=400,
        in_width=640,
        in_filter_size=(3, 3),
        # num_layers=4,  # Number of encoding blocks to use (typically 4 for enc_block1 to enc_block4)
        top_k=1.0,
        prune_ratio=0.0,
        target_prune_ratio=0.5,
        prune_step=0.05,
        score_method="feature_map",
    ):
        super(DeepVOGFoveated, self).__init__()

        # # Initialize DeepVOG backbone
        # self.backbone = DeepVOG_net(
        #     input_shape=(in_height, in_width, in_channels),
        #     filter_size=in_filter_size
        # )

        filter_size = in_filter_size
        input_shape = (in_height, in_width, in_channels)
        
        self.input_shape = input_shape
        self.filter_size = filter_size
        
        # Encoding Stream
        self.enc_block1 = EncodingBlock(in_channels=3, filter_size=filter_size, filters_num=16, layer_num=1, block_type="down", stage=1)
        self.enc_block2 = EncodingBlock(in_channels=32, filter_size=filter_size, filters_num=32, layer_num=1, block_type="down", stage=2)
        self.enc_block3 = EncodingBlock(in_channels=64, filter_size=filter_size, filters_num=64, layer_num=1, block_type="down", stage=3)
        self.enc_block4 = EncodingBlock(in_channels=128, filter_size=filter_size, filters_num=128, layer_num=1, block_type="down", stage=4)
        
        # Decoding Stream
        self.dec_block1 = DecodingBlock(in_channels=256, concat_channels=128, filter_size=filter_size, filters_num=256, layer_num=1, block_type="up", stage=1, stride=1)
        self.dec_block2 = DecodingBlock(in_channels=256, concat_channels=64, filter_size=filter_size, filters_num=256, layer_num=1, block_type="up", stage=2, stride=1)
        self.dec_block3 = DecodingBlock(in_channels=256, concat_channels=32, filter_size=filter_size, filters_num=128, layer_num=1, block_type="up", stage=3, stride=1)
        self.dec_block4 = DecodingBlock(in_channels=128, concat_channels=16, filter_size=filter_size, filters_num=64, layer_num=1, block_type="up", stage=4, stride=1)
        self.dec_block5 = DecodingBlock(in_channels=64, concat_channels=0, filter_size=filter_size, filters_num=32, layer_num=1, block_type="up", stage=5, stride=1, up_sampling=False)
        
        # Output layer
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1,1), stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        

         # Define fully connected layers for each exit point
        backbone_output_dim_enc = 256  # After enc_block4
        backbone_output_dim_dec_1 = 256  # After dec_block1
        backbone_output_dim_dec_2 = 256  # After dec_block2
        backbone_output_dim_dec_3 = 128  # After dec_block3
        backbone_output_dim_dec_4 = 64  # After dec_block4
        backbone_output_dim_dec = 32   # After dec_block5
        backbone_output_dim_out = 3    # After conv_out

        # Exit after Encoding Stream
        self.fc_enc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_output_dim_enc, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        # Exit after Decoding Streams
        self.fc_dec_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_output_dim_dec_1, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        self.fc_dec_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_output_dim_dec_2, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        self.fc_dec_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_output_dim_dec_3, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        self.fc_dec_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_output_dim_dec_4, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )


        self.fc_dec = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_output_dim_dec, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        # Exit after Output Layer
        # self.fc_out = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(backbone_output_dim_out, 512),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(512, 256),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(256, 128),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(128, 2)
        # )

        # Pruning and Scoring Parameters
        # self.top_k = top_k
        # self.score_method = score_method
        # self.prune_ratio = prune_ratio
        # self.prune_step = prune_step
        # self.target_prune_ratio = target_prune_ratio

    def forward(self, X):

        # input image is only 1 channel, so we need to repeat it to 3 channels
        X = X.repeat(1, 3, 1, 1)

        gaze_outputs = []
        
        # Encoding Stream
        X_jump1, X_out = self.enc_block1(X)
        X_jump2, X_out = self.enc_block2(X_out)
        X_jump3, X_out = self.enc_block3(X_out)
        X_jump4, X_out = self.enc_block4(X_out)


        gaze_enc = self.fc_enc(X_out)
        gaze_outputs.append(gaze_enc)

        
        # Decoding Stream
        X_out = self.dec_block1(X_out, X_jump4)    

        gaze_dec_1 = self.fc_dec_1(X_out)
        gaze_outputs.append(gaze_dec_1)

        X_out = self.dec_block2(X_out, X_jump3)

        gaze_dec_2 = self.fc_dec_2(X_out)
        gaze_outputs.append(gaze_dec_2)

        X_out = self.dec_block3(X_out, X_jump2)

        gaze_dec_3 = self.fc_dec_3(X_out)
        gaze_outputs.append(gaze_dec_3)

        X_out = self.dec_block4(X_out, X_jump1)

        gaze_dec_4 = self.fc_dec_4(X_out)
        gaze_outputs.append(gaze_dec_4)

        X_out = self.dec_block5(X_out, None)
        

        gaze_dec = self.fc_dec(X_out)
        gaze_outputs.append(gaze_dec)

        # # Output layer
        # X_out = self.conv_out(X_out)
        # X_out = self.softmax(X_out)

        # gaze_out = self.fc_out(X_out)
        # gaze_outputs.append(gaze_out)
        
        # Stack all gaze outputs
        gaze_outputs = torch.stack(gaze_outputs) # Shape: [6, batch_size, 2]

        return gaze_outputs

    def forward_timer(self, X, starters=None, enders=None):
        if starters is None or enders is None:
            raise ValueError("starters and enders must be provided for timing")
        
        X = X.repeat(1, 3, 1, 1)

        starters[0].record()  

        # Encoding Stream 
        X_jump1, X_out = self.enc_block1(X)
        X_jump2, X_out = self.enc_block2(X_out)
        X_jump3, X_out = self.enc_block3(X_out)
        X_jump4, X_out = self.enc_block4(X_out)
        # Decoding Stream
        X_out = self.dec_block1(X_out, X_jump4)      
        X_out = self.dec_block2(X_out, X_jump3)

        enders[0].record()   

        starters[1].record()
        
        X_out = self.dec_block3(X_out, X_jump2)
        X_out = self.dec_block4(X_out, X_jump1)
        X_out = self.dec_block5(X_out, None)

        enders[1].record()

        starters[2].record()

        
        # Output layer
        X_out = self.conv_out(X_out)
        X_out = self.softmax(X_out)

        gaze_out = self.fc_out(X_out)

        enders[2].record()

        return gaze_out
        

        
    
    # def forward_legacy(self, X):

    #     # input image is only 1 channel, so we need to repeat it to 3 channels
    #     X = X.repeat(1, 3, 1, 1)

    #     gaze_outputs = []
        
    #     # Encoding Stream
    #     X_jump1, X_out = self.backbone.enc_block1(X)
    #     X_jump2, X_out = self.backbone.enc_block2(X_out)
    #     X_jump3, X_out = self.backbone.enc_block3(X_out)
    #     X_jump4, X_out = self.backbone.enc_block4(X_out)

    #     # print("X_out shape:", X_out.shape)
    #     # print("X_jump4 shape:", X_jump4.shape)
    #     # print("X_jump3 shape:", X_jump3.shape)
    #     # print("X_jump2 shape:", X_jump2.shape)
    #     # print("X_jump1 shape:", X_jump1.shape)

    #     gaze_enc = self.fc_enc(X_out)
    #     # print("gaze_enc shape:", gaze_enc.shape)
    #     gaze_outputs.append(gaze_enc)

        
    #     # Decoding Stream
    #     X_out = self.backbone.dec_block1(X_out, X_jump4)      # No skip connection for the first decoding block?
        
    #     # print("X_out shape:", X_out.shape)
    #     # print("X_jump4 shape:", X_jump4.shape)

    #     X_out = self.backbone.dec_block2(X_out, X_jump3)
    #     X_out = self.backbone.dec_block3(X_out, X_jump2)
    #     X_out = self.backbone.dec_block4(X_out, X_jump1)
    #     X_out = self.backbone.dec_block5(X_out, None)
        


    #     gaze_dec = self.fc_dec(X_out)
    #     # print("gaze_dec shape:", gaze_dec.shape)
    #     gaze_outputs.append(gaze_dec)

    #     # Output layer
    #     X_out = self.backbone.conv_out(X_out)
    #     X_out = self.backbone.softmax(X_out)

    #     gaze_out = self.fc_out(X_out)
    #     # print("gaze_out shape:", gaze_out.shape)
    #     gaze_outputs.append(gaze_out)
        
    #     # Stack all gaze outputs
    #     gaze_outputs = torch.stack(gaze_outputs)  # Shape: [3, batch_size, 2]
    #     # print("gaze_outputs shape:", gaze_outputs.shape)

    #     return gaze_outputs

        


    # def forward_timer(self, x, starters=None, enders=None):
    #     if starters is None or enders is None:
    #         raise ValueError("starters and enders must be provided for timing")

    #     enc_outputs = []
    #     gaze_outputs = []

    #     # Encoding Stream with Timing
    #     for idx, layer in enumerate(self.encoding_layers):
    #         starters[idx].record()  # Start timing for this encoding block
    #         x, _ = layer(x)
    #         enders[idx].record()    # End timing for this encoding block
    #         enc_outputs.append(x)

    #     # Exit Point 1: After Encoding Stream
    #     feat_enc = enc_outputs[-1]
    #     starters[len(self.encoding_layers)].record()  # Start timing for FC_enc
    #     gaze_enc = self.fc_enc(feat_enc)
    #     enders[len(self.encoding_layers)].record()    # End timing for FC_enc
    #     gaze_outputs.append(gaze_enc)

    #     # Decoding Stream with Timing
    #     skip_connections = enc_outputs[:-1][::-1]  # Reverse order for skip connections
    #     for idx, layer in enumerate(self.decoding_layers):
    #         starters[len(self.encoding_layers) + 1 + idx].record()  # Start timing for this decoding block
    #         if idx < len(skip_connections):
    #             x = layer(x, skip_connections[idx])
    #         else:
    #             x = layer(x, None)
    #         enders[len(self.encoding_layers) + 1 + idx].record()    # End timing for this decoding block
    #         dec_outputs = x  # Keep the latest decoding output

    #     # Exit Point 2: After Decoding Stream
    #     feat_dec = dec_outputs
    #     starters[len(self.encoding_layers) + 1 + len(self.decoding_layers)].record()  # Start timing for FC_dec
    #     gaze_dec = self.fc_dec(feat_dec)
    #     enders[len(self.encoding_layers) + 1 + len(self.decoding_layers)].record()    # End timing for FC_dec
    #     gaze_outputs.append(gaze_dec)

    #     # Output Layer with Timing
    #     starters[-1].record()  # Start timing for output layer
    #     x = self.output_layer(x)
    #     x = self.softmax(x)
    #     enders[-1].record()    # End timing for output layer

    #     # Exit Point 3: After Output Layer
    #     feat_out = x
    #     starters[-2].record()  # Start timing for FC_out
    #     gaze_out = self.fc_out(feat_out)
    #     enders[-2].record()    # End timing for FC_out
    #     gaze_outputs.append(gaze_out)

    #     # Stack all gaze outputs
    #     gaze_outputs = torch.stack(gaze_outputs, dim=1)  # Shape: [batch_size, 3, 2]

    #     return gaze_outputs




# def load_DeepVOG(model_path="DeepVOG_weights.pth"):
#     model = DeepVOG_net(input_shape=(240, 320, 3), filter_size=(10,10))
#     model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     model.eval()
#     return model



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