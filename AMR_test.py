# when rendering, we can do adaptive mesh refinement to save time and memory
# for instance, we can count the number of gaussians/points on each tile as a creteria
# if the number is less than a threshold, we assign a lower refinement level 
# when rendering lower level tiles, we can render only a few points acurately and interpolate the rest
# then when the refinement level increases, we can still reuse the acurate points, and only render the rest

from scipy import interpolate
import numpy as np

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


tile_level = 4 # 2^tile_level is the number of pixels in a tile
coarse_level = 2 # 2^coarse_level is the number of acurate pixels in a coarse pixel
camera_idx = 0 # idx in all cameras

# same args as render.py
parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
args = get_combined_args(parser)
print("Rendering " + args.model_path)
safe_state(args.quiet)
mydataset = model.extract(args)

# render the acurate image for reference
gaussians = GaussianModel(mydataset.sh_degree) # create an empty gaussian model
scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False) # load the model and cameras
view = scene.getTrainCameras()[camera_idx]
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
rendering = render(view, gaussians, pipeline, background)["render"]
torchvision.utils.save_image(rendering, "original_render.png")

# compare with an image purely on a coarse level
skip_lelvel = tile_level - coarse_level # skip rendering the lower level tiles, i.e. the accurate pixels are defined every 2^skip_level pixels in x,y directions
accurate_x = np.array(range(0,view.image_height,2**skip_lelvel)) # note the image is transposed compared with usual convention
accurate_y = np.array(range(0,view.image_width,2**skip_lelvel))
# I will use interpolate.griddata to interpolate the accurate points to the whole image
# input is a list of (x,y) pairs, get them from accurate_x and accurate_y
accurate_points = np.array([(x,y) for y in accurate_y for x in accurate_x ])
all_points = np.array([(x,y) for y in range(view.image_width) for x in range(view.image_height)])
# rendering is torch tensor of shape (3, image_height, image_width), use interpolate.griddata to interpolate the accurate points
# first interpolate the red channel
red = rendering[0].cpu().detach().numpy()
accurate_red = np.array([red[x,y] for x,y in accurate_points])
red_interpolated = interpolate.griddata(accurate_points, accurate_red, all_points, method='cubic')
# then interpolate the green channel
green = rendering[1].cpu().detach().numpy()
accurate_green = np.array([green[x,y] for x,y in accurate_points])
green_interpolated = interpolate.griddata(accurate_points, accurate_green, all_points, method='cubic')
# then interpolate the blue channel
blue = rendering[2].cpu().detach().numpy()
accurate_blue = np.array([blue[x,y] for x,y in accurate_points])
blue_interpolated = interpolate.griddata(accurate_points, accurate_blue, all_points, method='cubic')
# combine the three channels
interpolated_rendering = torch.tensor([red_interpolated, green_interpolated, blue_interpolated], dtype=torch.float32, device="cuda")
# reshape to (3, image_width, image_height)
interpolated_rendering = interpolated_rendering.reshape(3, view.image_width, view.image_height)
# transpose the last two dimensions to get the usual convention
interpolated_rendering = interpolated_rendering.permute(0,2,1)
torchvision.utils.save_image(interpolated_rendering, "coarse_render.png")


# the above gives a coarse rendering globally, we can also do AMR to decide the coarse_level of each tile

# first transform the gaussians._xyz to the image space, refer to the cuda code

    # // Transform point by projecting
    # float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    # float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    # float p_w = 1.0f / (p_hom.w + 0.0000001f);
    # float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

