# from mpi4py import MPI
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import time
import json
# from gaussian_renderer_amr import render_mpi, GaussianModel

import math
from diff_gaussian_rasterization_amr import GaussianRasterizationSettings, GaussianRasterizer

from diff_gaussian_rasterization_amr import _RasterizeGaussians

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
# from fovealnet.timm_vit import VisionTransformer

def load_image(image_path):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

pix_x = 1920
pix_y = 1080


# Define number of images and steps
num_images = 50  # Adjust based on your data
max_steps = 5  # Number of fovea steps


parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

# Add arguments from track.py
parser.add_argument("--foveal_model_path", default="/home/ubuntu/gaussian_splatting_with_eye_tracking/fovealnet/results_20241022/model_minmax_0.8.pt", type=str)
parser.add_argument("--eye_image_folder", default="/home/ubuntu/openeds/test/sequences/0000/", type=str)
parser.add_argument("--eye_image_sequence_folder", default="/home/ubuntu/openeds/test/sequences/", type=str)
parser.add_argument("--eye_image_sequence_id_start", default=None, type=int)
parser.add_argument("--eye_image_sequence_id_end", default=None, type=int)
parser.add_argument("--foveal_output_file", type=str, default="predictions.txt")
parser.add_argument("--foveal_layer_timer", action="store_true")

parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--show_fps", action="store_true")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--test_no_render_laststep", action="store_true") # test the time of purely passing the data in foveastep 4
parser.add_argument("--angle_to_pix_radius", default=600.0, type=float) # a factor (radius) to convert pitch and yaw angles to pixel position of the gaze
parser.add_argument("--gaze_r2", default=600, type=float)
parser.add_argument("--gaze_r3", default=400, type=float)
parser.add_argument("--gaze_r4", default=200, type=float)
# parser.add_argument("--no_competing_device", action="store_true") # do not let the 3DGS and fovealnet process to compete for the GPU resource
args = get_combined_args(parser)


# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# Create shared memory window for gaze predictions
# if rank == 0:
#     # Rank 0 creates the shared memory
#     sync_gaze_prediction = np.zeros(2, dtype=np.float32)
# else:
#     # Other ranks don't need local buffer
#     sync_gaze_prediction = np.zeros(2, dtype=np.float32)

# sync_gaze_prediction = np.zeros(2, dtype=np.float32)
# win = MPI.Win.Create(sync_gaze_prediction, comm=comm)

# gaze_predictions_buffer = np.zeros((150, 2), dtype=np.float32) # at most 150 gaze for each head position (50 in the test set) 
# fovealnet_level_buffer = np.zeros(150, dtype=np.int32) # the level of fovealnet inference for each eye image (gaze)
# GS_level_buffer = np.zeros(150, dtype=np.int32) # the level of 3DGS rendering for each eye image (gaze)


# local_predictions_buffer = np.zeros((150, 2), dtype=np.float32) 
# local_fovealnet_level_buffer = np.zeros(150, dtype=np.int32)
# local_GS_level_buffer = np.zeros(150, dtype=np.int32)

# win1 = MPI.Win.Create(gaze_predictions_buffer, comm=comm)
# win2 = MPI.Win.Create(fovealnet_level_buffer, comm=comm)
# win3 = MPI.Win.Create(GS_level_buffer, comm=comm)


# define vison transformer here in this script

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
    
    def forward_timer(self, x, starters=None, enders=None, img_idx=0):
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
        





# define render function in this script

def render_serial(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
                gaze_x = 0,  # gaze direction x
                gaze_y = 0,  # gaze direction y
                gaze_r2 = 1e4 , gaze_r3=1e4, gaze_r4=1e4,  # radii of the foveal level 2,3,4
                fovealnet : VisionTransformer = None, fovealnet_starters=None, fovealnet_enders=None, 
                sequence_folder=None, layer_timings_per_image=None, predictions=None, layer_times=None, inference_times=None,
                total_sequence_time = None, 
           starter=None,ender=None, starters=None, enders=None, steps_performed=None,
           interpolate_image = False, test_no_render_laststep=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    num_events = len(fovealnet.transformer_layers) + 2

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # print(raster_settings.debug)

    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rawrasterizer = _RasterizeGaussians

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # if starter is not None:
    #     starter.record()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
  
    # when using the raw rasterizer
    if shs is None:
        shs = torch.Tensor([])
    if colors_precomp is None:
        colors_precomp = torch.Tensor([])

    if scales is None:
        scales = torch.Tensor([])
    if rotations is None:
        rotations = torch.Tensor([])
    if cov3D_precomp is None:
        cov3D_precomp = torch.Tensor([])

    out_color_precomp = torch.Tensor([])



    if pipe.debug:
        print(" fovea step 0 ")
    foveaStep = 0
    buffered = False
    geomBuffer_precomp = torch.Tensor([]).to(torch.uint8)
    binningBuffer_precomp = torch.Tensor([]).to(torch.uint8)
    imageBuffer_precomp = torch.Tensor([]).to(torch.uint8)

    if starter is not None:
        starter.record()

    if steps_performed is not None:
        steps_performed[0] = 1
    if starters is not None:
        starters[0].record()


    # step 0: compute only the buffers
    rendered_image0, radii, geomBuffer, binningBuffer, imageBuffer = rawrasterizer.apply(
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
            foveaStep,
            gaze_x,  # gaze direction x
            gaze_y,  # gaze direction y
            gaze_r2, gaze_r3, gaze_r4,  # radii of the foveal level 2,3,4
            out_color_precomp,
            # radii_precomp,
            # means2D_precomp,
            # conic_opacity_precomp,
            # geom_rgb_precomp,
            # point_list_precomp,
            # ranges_precomp,
            # tile_AMR_levels_last,
            # tile_AMR_levels_current,
            geomBuffer_precomp,
            binningBuffer_precomp,
            imageBuffer_precomp,
            False, # interpolate_image
        raster_settings,
    )
        
    if enders is not None:
        enders[0].record()

    if pipe.debug:
        reds = rendered_image0[0].cpu().detach().numpy()
        redmask = reds == 0
        print("Skipped pixels: ", np.sum(redmask))
        print("Total pixels: ", redmask.size) 
        torchvision.utils.save_image(rendered_image0, "tmp0.png")


    foveaStep = 1
    out_color_precomp = rendered_image0 # should be all 0
    geomBuffer_precomp = geomBuffer
    binningBuffer_precomp = binningBuffer
    imageBuffer_precomp = imageBuffer
    if pipe.debug:
        reds = out_color_precomp[0].cpu().detach().numpy()
        redmask = reds == 0
        print("combined, Skipped pixels: ", np.sum(redmask))
        print("combined, Total pixels: ", redmask.size) 
    
    if pipe.debug:
        print(" fovea step 1 ")


    if steps_performed is not None:
        steps_performed[1] = 1
    if starters is not None:
        starters[1].record()

    # step 1: compute the lowest quality
    rendered_image1, _, geomBuffer, binningBuffer, imageBuffer = rawrasterizer.apply(
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
            foveaStep,
            gaze_x,  # gaze direction x
            gaze_y,  # gaze direction y
            gaze_r2, gaze_r3, gaze_r4,  # radii of the foveal level 2,3,4
            out_color_precomp,
            # radii_precomp,
            # means2D_precomp,
            # conic_opacity_precomp,
            # geom_rgb_precomp,
            # point_list_precomp,
            # ranges_precomp,
            # tile_AMR_levels_last,
            # tile_AMR_levels_current,
            geomBuffer_precomp,
            binningBuffer_precomp,
            imageBuffer_precomp,
            False, # interpolate_image
        raster_settings,
    )
        
    if enders is not None:
        enders[1].record()

    if pipe.debug:
        reds = rendered_image1[0].cpu().detach().numpy()
        redmask = reds == 0
        print("Skipped pixels: ", np.sum(redmask))
        print("Total pixels: ", redmask.size) 
        torchvision.utils.save_image(rendered_image1, "tmp1.png")
    out_color_precomp = out_color_precomp + rendered_image1

    # co-design with fovealnet
    cur_gaze_dir = np.zeros(2)
    # for img_idx in range(50):
    # num_images = 0
    img_idx = -1
    for image_name in os.listdir(sequence_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(sequence_folder, image_name)
            image = load_image(image_path).to(device)
            img_idx += 1

            x = image
            print("inference fovealnet prediction for eye image index: ", img_idx)

            # record also the total time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            # loop over vision transformer layers
            fovealnet_starters[0].record()  # Start timing for patch embedding
            x = fovealnet.backbone.patch_embed(x)
            if fovealnet.backbone.pos_embed.shape[1] == 197 and x.shape[1] == 196:
                pos_embed = fovealnet.backbone.pos_embed[:, 1:, :] 
            else:
                pos_embed = fovealnet.backbone.pos_embed
            x = fovealnet.backbone.pos_drop(x + pos_embed)
            fovealnet_enders[0].record()  # End timing for patch embedding

            
            for i, block in enumerate(fovealnet.transformer_layers):
                fovealnet_starters[i+1].record()  # Start timing for this transformer block
                x = block(x)
                if i % 2 == 1 and fovealnet.score_method == "attention":
                    attn_scores = fovealnet.attention_scores.mean(dim=-1)
                    topk_indices = attn_scores.topk(
                        int(fovealnet.top_k * attn_scores.size(1)), dim=1, largest=True
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
                    gaze_dir = F.relu(fovealnet.fc1(features))
                    gaze_dir = F.relu(fovealnet.fc2(gaze_dir))
                    gaze_dir = F.relu(fovealnet.fc3(gaze_dir))
                    gaze_dir = fovealnet.fc4(gaze_dir)
                    # update the foveal net step for rendering
                    cur_gaze_dir = gaze_dir.cpu().numpy()[0] 
                    foveaStep = ((i+1) // 2) + 1
                fovealnet_enders[i+1].record()  # End timing for this transformer block
                # render stuff at i=1,3,5
                if i%2 == 1:
                    mid_x = (pix_x*ratio) / 2
                    mid_y = (pix_y*ratio) / 2
                    gaze_x = np.sin(cur_gaze_dir[0]) * args.angle_to_pix_radius + mid_x
                    gaze_y = np.sin(cur_gaze_dir[1]) * args.angle_to_pix_radius + mid_y

                    print(f"gaze at pixel: {gaze_x}, {gaze_y}")

                    if steps_performed is not None:
                        steps_performed[foveaStep + img_idx*5] = 1

                    if starters is not None:
                        starters[foveaStep + img_idx*5].record()

                    geomBuffer_precomp = geomBuffer
                    binningBuffer_precomp = binningBuffer
                    imageBuffer_precomp = imageBuffer
                    rendered_image0, radii, geomBuffer, binningBuffer, imageBuffer = rawrasterizer.apply(
                        means3D,
                        means2D,
                        shs,
                        colors_precomp,
                        opacity,
                        scales,
                        rotations,
                        cov3D_precomp,
                            foveaStep,
                            gaze_x,  # gaze direction x
                            gaze_y,  # gaze direction y
                            gaze_r2,gaze_r3,gaze_r4,  # radii of the foveal level 2,3,4
                            out_color_precomp,
                            # radii_precomp,
                            # means2D_precomp,
                            # conic_opacity_precomp,
                            # geom_rgb_precomp,
                            # point_list_precomp,
                            # ranges_precomp,
                            # tile_AMR_levels_last,
                            # tile_AMR_levels_current,
                            geomBuffer_precomp,
                            binningBuffer_precomp,
                            imageBuffer_precomp,
                            False, # interpolate_image
                        raster_settings,
                    )
                    out_color_precomp = out_color_precomp + rendered_image0
                    if enders is not None:
                        enders[foveaStep + img_idx*5].record()


                

            fovealnet_starters[len(fovealnet.transformer_layers)+1].record()  # Start timing for final layers
            features = x.mean(dim=1)
            gaze_dir = F.relu(fovealnet.fc1(features))
            gaze_dir = F.relu(fovealnet.fc2(gaze_dir))
            gaze_dir = F.relu(fovealnet.fc3(gaze_dir))
            gaze_dir = fovealnet.fc4(gaze_dir)
            fovealnet_enders[len(fovealnet.transformer_layers)+1].record()  # End timing for final layers


            output = gaze_dir.cpu().numpy()[0]

            
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            total_sequence_time += elapsed_time
            # num_images += 1
            inference_times.append(elapsed_time)

            if args.foveal_layer_timer:
                # Record layer timings for this image
                layer_timings = []
                for i in range(num_events):
                    layer_time = fovealnet_starters[i].elapsed_time(fovealnet_enders[i])
                    layer_times[i] += layer_time
                    layer_timings.append(layer_time)
                layer_timings_per_image.append(layer_timings)
            

            print(f"foveal net time for image idx {img_idx}: {elapsed_time:.2f} ms")
            print("By layer:")
            for i in range(len(layer_times)):
                if i==0:
                    print(f"Layer {i} (embedding): {fovealnet_starters[i].elapsed_time(fovealnet_enders[i]):.2f} ms")
                elif i==len(layer_times)-1:
                    print(f"Layer {i} (final layer): {fovealnet_starters[i].elapsed_time(fovealnet_enders[i]):.2f} ms")
                else:
                    print(f"Layer {i} (transformer block): {fovealnet_starters[i].elapsed_time(fovealnet_enders[i]):.2f} ms")
            prediction = output
            predictions.append((image_name, prediction))


    if ender is not None:
        ender.record()


    average_time = total_sequence_time / num_images
    print(f"Average inference time for sequence {seq_id}: {average_time:.2f} ms")

    if args.foveal_layer_timer:
        print("\nLayer-wise timing:")
        print(f"Patch embedding time: {layer_times[0]/num_images:.2f} ms")
        for i, time in enumerate(layer_times[1:-1], 1):
            print(f"Transformer block {i} time: {time/num_images:.2f} ms")
        print(f"Final layers time: {layer_times[-1]/num_images:.2f} ms")

    # Write predictions to output file
    output_file = f"predictions_{seq_id}.txt"
    with open(output_file, 'w') as f:
        for image_name, prediction in predictions:
            f.write(f"{image_name}: Pitch={prediction[0]:.4f}, Yaw={prediction[1]:.4f}\n")



    return {"render": out_color_precomp,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            # "means2D": parsed_means2D,
            # "conic_opacity": parsed_conic_opacity,
            # "geom_rgb": parsed_geom_rgb,
            # "point_list": parsed_point_list,
            # "ranges": parsed_ranges,
            # "tile_AMR_levels": parsed_tile_AMR_levels
            }



# Initialize lists to store timing data
total_times = []
image_step_times = []

# set a prefix ("[3DGS]") for all print()
# print = lambda x: print("[3DGS]", x)
# Replace the lambda definition with this
# print = lambda *args, **kwargs: __builtins__.print("[3DGS]", *args, **kwargs)

device = torch.device("cuda")
props = torch.cuda.get_device_properties(device)
num_sms = props.multi_processor_count
print(f"Gaussian Splatting process is using {num_sms} stream multiprocessors.")

print("Rendering " + args.model_path)
safe_state(args.quiet)
mydataset = model.extract(args)

gaussians = GaussianModel(mydataset.sh_degree)
scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False)
views = scene.getTrainCameras()
views = [views[i] for i in range(100)]
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

ratio = 1

print(f"Rendering at {int(pix_x*ratio)}x{int(pix_y*ratio)}")
# change the image width and height
for i in range(len(views)):
    views[i].image_width = int(pix_x*ratio)
    views[i].image_height = int(pix_y*ratio)


# test fps by counting time
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# record 4 fov steps separately
# starter0, ender0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# starter2, ender2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# starter3, ender3 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# starter4, ender4 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

mystarters = []
myenders = []
for imgidx in range(50):
    for stepidx in range(5):
        mystarters.append(torch.cuda.Event(enable_timing=True))
        myenders.append(torch.cuda.Event(enable_timing=True))


# load fovealnet model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
props = torch.cuda.get_device_properties(device)
num_sms = props.multi_processor_count
print(f"FovealNet process is using {num_sms} stream multiprocessors.")


fovmodel = VisionTransformer(num_layers=6, top_k=1.0).to(device)
fovmodel.load_state_dict(torch.load(args.foveal_model_path, map_location=device))
fovmodel.eval()

# Initialize lists to store timing data
inference_times = []
layer_timings_per_image = []


torch.cuda.synchronize()

for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    seq_id = idx
    print(f"Rendering view {idx} in 3DGS, and using sequence {seq_id} for fovealnet")

    # Create CUDA events for timing if foveal_layer_timer is enabled
    if args.foveal_layer_timer:
        num_events = len(fovmodel.transformer_layers) + 2  # +1 for patch embedding, +1 for final layers
        fovealnet_starters = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
        fovealnet_enders = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
    else:
        fovealnet_starters, fovealnet_enders = None, None
    sequence_folder = os.path.join(args.eye_image_sequence_folder, f"{seq_id:04d}")
    if not os.path.exists(sequence_folder):
        print(f"Skipping sequence {seq_id} as it does not exist")
        continue

    # Process images and make predictions
    predictions = []
    total_sequence_time = 0
    layer_times = [0] * (len(fovmodel.transformer_layers) + 2) if args.foveal_layer_timer else None

    

    timeall = 0
    # time0 = 0
    # time1 = 0
    # time2 = 0
    # time3 = 0
    # time4 = 0

    
    steps_performed = [0] * 250
    with torch.no_grad():
        rendering = render_serial(view, gaussians, pipeline, background,starter = starter, ender= ender, 
                                starters = mystarters, enders = myenders, steps_performed=steps_performed,
                                gaze_r2=args.gaze_r2, gaze_r3=args.gaze_r3, gaze_r4=args.gaze_r4,
                                fovealnet=fovmodel, fovealnet_starters=fovealnet_starters, fovealnet_enders=fovealnet_enders,
                                sequence_folder=sequence_folder, layer_timings_per_image=layer_timings_per_image,
                                predictions=predictions, layer_times=layer_times, inference_times=inference_times,
                                total_sequence_time=total_sequence_time, 
                                test_no_render_laststep=args.test_no_render_laststep
                                )["render"]
        
    torch.cuda.synchronize()
    timeall += starter.elapsed_time(ender)
    total_times.append(timeall)
    # time0 += starter0.elapsed_time(ender0)
    # time1 += starter1.elapsed_time(ender1)
    # time2 += starter2.elapsed_time(ender2)
    # time3 += starter3.elapsed_time(ender3)
    # time4 += starter4.elapsed_time(ender4)
    # print(f"Rendering time for view {idx}: {timeall:.2f} ms")
    # print(f"Rendering time for view {idx} step 0: {time0:.2f} ms")
    # print(f"Rendering time for view {idx} step 1: {time1:.2f} ms")
    # print(f"Rendering time for view {idx} step 2: {time2:.2f} ms")
    # print(f"Rendering time for view {idx} step 3: {time3:.2f} ms")
    # print(f"Rendering time for view {idx} step 4: {time4:.2f} ms")
    print(f"Total Rendering time for view {idx}: {timeall:.2f} ms")


    # Collect step times for each image
    view_image_step_times = []
    for img_idx in range(num_images):
        img_step_times = []
        for step_idx in range(max_steps):
            idx_flat = step_idx + img_idx * max_steps
            if steps_performed[idx_flat] == 1:
                step_time = mystarters[idx_flat].elapsed_time(myenders[idx_flat])
                img_step_times.append(step_time)
                print(f"Rendering time for view {idx}, step {step_idx} for eye image {img_idx}: {step_time:.2f} ms")
            else:
                img_step_times.append(0.0)  # If step wasn't performed, record 0
        view_image_step_times.append(img_step_times)
    image_step_times.append(view_image_step_times)

# Save timing data to a file
with open('timing_data.json', 'w') as f:
    json.dump({
        'total_times': total_times,
        'image_step_times': image_step_times
    }, f)


with open('fovealnet_timing_data.json', 'w') as f:
    json.dump({
        'inference_times': inference_times,
        'layer_timings_per_image': layer_timings_per_image
    }, f)

