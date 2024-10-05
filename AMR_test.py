# when rendering, we can do adaptive mesh refinement to save time and memory
# for instance, we can count the number of gaussians/points on each tile as a creteria
# if the number is less than a threshold, we assign a lower refinement level 
# when rendering lower level tiles, we can render only a few points acurately and interpolate the rest
# then when the refinement level increases, we can still reuse the acurate points, and only render the rest

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import product


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
# coarse_level = 3 # 2^coarse_level is the number of acurate pixels in a coarse pixel
camera_idx = 0 # idx in all cameras
AMR_factor = 1.2 # the factor to determine the AMR level, the higher the factor, the higher the AMR level overall

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
torchvision.utils.save_image(rendering, "original_render_view%d.png"%camera_idx)

# compare with an image purely on a coarse level
# skip_lelvel = tile_level - coarse_level # skip rendering the lower level tiles, i.e. the accurate pixels are defined every 2^skip_level pixels in x,y directions
# accurate_x = np.array(range(0,view.image_height,2**skip_lelvel)) # note the image is transposed compared with usual convention
# accurate_y = np.array(range(0,view.image_width,2**skip_lelvel))
# # I will use interpolate.griddata to interpolate the accurate points to the whole image
# # input is a list of (x,y) pairs, get them from accurate_x and accurate_y
# accurate_points = np.array([(x,y) for y in accurate_y for x in accurate_x ])
# all_points = np.array([(x,y) for y in range(view.image_width) for x in range(view.image_height)])
# # rendering is torch tensor of shape (3, image_height, image_width), use interpolate.griddata to interpolate the accurate points
# # first interpolate the red channel
# red = rendering[0].cpu().detach().numpy()
# accurate_red = np.array([red[x,y] for x,y in accurate_points])
# red_interpolated = interpolate.griddata(accurate_points, accurate_red, all_points, method='cubic')
# # then interpolate the green channel
# green = rendering[1].cpu().detach().numpy()
# accurate_green = np.array([green[x,y] for x,y in accurate_points])
# green_interpolated = interpolate.griddata(accurate_points, accurate_green, all_points, method='cubic')
# # then interpolate the blue channel
# blue = rendering[2].cpu().detach().numpy()
# accurate_blue = np.array([blue[x,y] for x,y in accurate_points])
# blue_interpolated = interpolate.griddata(accurate_points, accurate_blue, all_points, method='cubic')
# # combine the three channels
# interpolated_rendering = torch.tensor([red_interpolated, green_interpolated, blue_interpolated], dtype=torch.float32, device="cuda")
# # reshape to (3, image_width, image_height)
# interpolated_rendering = interpolated_rendering.reshape(3, view.image_width, view.image_height)
# # transpose the last two dimensions to get the usual convention
# interpolated_rendering = interpolated_rendering.permute(0,2,1)

# torchvision.utils.save_image(interpolated_rendering, "coarse_render.png")


# the above gives a coarse rendering globally, we can also do AMR to decide the coarse_level of each tile

# first transform the gaussians._xyz to the image space, refer to the cuda code

    # // Transform point by projecting
    # float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    # float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    # float p_w = 1.0f / (p_hom.w + 0.0000001f);
    # float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    # float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };


    # __forceinline__ __device__ float ndc2Pix(float v, int S)
    # {
    # 	return ((v + 1.0) * S - 1.0) * 0.5;
    # }

def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

# this operation is the same as geom_transform_points utils/graphics_utils.py
from utils.graphics_utils import geom_transform_points
# this is the projection matrix passed to  refer to gaussian_renderer/__init__.py:44
projM = view.full_proj_transform
# refer to gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h:152, also filter by p_view
# input viewmatrix is world_view_transform according to gaussian-splatting/gaussian_renderer/__init__.py
viewM = view.world_view_transform
# similar to geom_transform_points
ones = torch.ones(gaussians._xyz.shape[0], 1, dtype=gaussians._xyz.dtype, device=gaussians._xyz.device)
points_hom = torch.cat([gaussians._xyz, ones], dim=1)
points_view = torch.matmul(points_hom, viewM.unsqueeze(0))[0]
points_view = points_view.cpu().detach().numpy()


gaussians_proj = geom_transform_points(gaussians._xyz, projM)
gaussians_proj = gaussians_proj.cpu().detach().numpy()
gaussians_2d_x = ndc2Pix(gaussians_proj[:,0], view.image_width)
gaussians_2d_y = ndc2Pix(gaussians_proj[:,1], view.image_height)
mask = (gaussians_2d_x >= 0) & (
    gaussians_2d_x < view.image_width) & (
        gaussians_2d_y >= 0) & (
            gaussians_2d_y < view.image_height) & (
                points_view[:,2] > 0.2) # only keep the points in front of the camera, refer to gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h:154,
gaussians_2d_x = gaussians_2d_x[mask]
gaussians_2d_y = gaussians_2d_y[mask]

plt.figure().set_size_inches(10,6)
plt.imshow(rendering.cpu().detach().numpy().transpose(1,2,0))
# put the mesh on top of the image
plt.axhline(0, color='r', linewidth=3, alpha=0.5, label="tiles")
for x in range(0,view.image_height,2**tile_level):
    plt.axhline(x, color='r', linewidth=0.5, alpha=0.5)


for y in range(0,view.image_width,2**tile_level):
    plt.axvline(y, color='r', linewidth=0.5, alpha=0.5)

# scatter the gaussians
plt.scatter(gaussians_2d_x[0], gaussians_2d_y[0], marker='.', color='b', s=20, alpha=1, label="gaussians") 
plt.scatter(gaussians_2d_x, gaussians_2d_y, marker='.', color='b', s=0.05, alpha=0.2)

plt.xlabel("horizontal pixel", fontsize=16)
plt.ylabel("vertical pixel", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.legend(fontsize=16)
# pute legend at top (like a title)
plt.legend(loc='upper center', bbox_to_anchor=(0.75, 1.15), ncol=2, fontsize=16)
plt.xlim(-1,view.image_width)
plt.ylim(-1,view.image_height)
plt.gca().invert_yaxis()
plt.savefig("original_render_all_gaussians_view%d.png"%camera_idx)

# now count the num of gaussians on each tile
tiles_num_x = view.image_width // 2**tile_level + 1 # from 0 to view.image_width//2**tile_level
tiles_num_y = view.image_height // 2**tile_level + 1

gaussians_2d_tilex = (gaussians_2d_x // 2**tile_level).astype(int)
gaussians_2d_tiley = (gaussians_2d_y // 2**tile_level).astype(int)

tiles_gaussian_count = np.zeros((tiles_num_x, tiles_num_y))
tiles_AMRlevel = np.zeros((tiles_num_x, tiles_num_y)) 

# Create a new image to overlay the tile counts
tile_count_image = np.zeros((view.image_height, view.image_width)) +1 # avoid 0

for x in range(tiles_num_x):
    for y in range(tiles_num_y):
        mask = (gaussians_2d_tilex == x) & (gaussians_2d_tiley == y)
        tiles_gaussian_count[x,y] = np.sum(mask)
        start_x = x * 2**tile_level
        end_x = min((x + 1) * 2**tile_level, view.image_width)
        start_y = y * 2**tile_level
        end_y = min((y + 1) * 2**tile_level, view.image_height)
        tile_count_image[start_y:end_y, start_x:end_x] = tiles_gaussian_count[x, y]

# tiles_AMRlevel = np.floor(np.log10(tiles_gaussian_count+1)).astype(int) +1 # a simple criterion, using log10(count) to determine the AMR level
# tiles_AMRlevel_image = np.floor(np.log10(tile_count_image+1)).astype(int) +1

tiles_AMRlevel = np.floor( AMR_factor* np.log10(tiles_gaussian_count+1)).astype(int) +1 # a simple criterion, using log10(count) to determine the AMR level
tiles_AMRlevel_image = np.floor(AMR_factor * np.log10(tile_count_image+1)).astype(int) +1

tiles_AMRlevel[tiles_AMRlevel > 4] = 4
tiles_AMRlevel_image[tiles_AMRlevel_image > 4] = 4

# now imshow the tiles_gaussian_count, but note that each pix for tiles_gaussian_count should cover 2^tile_level pixels in previous image
plt.figure().set_size_inches(10,6)
plt.imshow(rendering.cpu().detach().numpy().transpose(1,2,0))

# Overlay the tile count image with transparency and log scale
plt.imshow(np.log10(tile_count_image), cmap='jet',  alpha=0.5)


# Add colorbar to show the scale of gaussian counts
plt.colorbar(label='$\\log_{10}$ # of gaussians')

plt.xlabel("horizontal pixel", fontsize=16)
plt.ylabel("vertical pixel", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(-1,view.image_width)
plt.ylim(-1,view.image_height)
plt.gca().invert_yaxis()
plt.savefig("original_render_tile_gaussians_view%d.png"%camera_idx)


# now imshow the tiles_gaussian_count, but note that each pix for tiles_gaussian_count should cover 2^tile_level pixels in previous image
plt.figure().set_size_inches(10,6)
plt.imshow(rendering.cpu().detach().numpy().transpose(1,2,0))

# Overlay the tile count image with AMR level, note that the level are discrete numbers 1,2,3,4, adjust the cmap accordingly

# Define discrete colormap
cmap = ListedColormap(['blue', 'cyan', 'yellow', 'red'])
bounds = [1, 2, 3, 4, 5]
norm = BoundaryNorm(bounds, cmap.N)
plt.imshow(tiles_AMRlevel_image, cmap=cmap, norm=norm, alpha=0.5)

# Add colorbar to show the scale of AMR levels
plt.colorbar(label='AMR level', ticks=[1, 2, 3, 4])

plt.xlabel("horizontal pixel", fontsize=16)
plt.ylabel("vertical pixel", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(-1,view.image_width)
plt.ylim(-1,view.image_height)
plt.gca().invert_yaxis()
plt.savefig("original_render_tile_AMR_view%d.png"%camera_idx)

# now render the image with AMR
accurate_points = []
for x in range(tiles_num_x):
    for y in range(tiles_num_y):
        start_x = x * 2**tile_level
        end_x = min((x + 1) * 2**tile_level, view.image_width-1)
        start_y = y * 2**tile_level
        end_y = min((y + 1) * 2**tile_level, view.image_height-1)
        accurate_x = range(start_x,end_x,2**(tile_level - tiles_AMRlevel[x,y]))
        accurate_y = range(start_y,end_y,2**(tile_level - tiles_AMRlevel[x,y]))
        pixel_combinations = list(product(accurate_x, accurate_y))
        pixel_combinations = np.array(pixel_combinations)
        accurate_points.extend(pixel_combinations)

accurate_points = np.array(accurate_points)
all_points = list(product(range(view.image_width), range(view.image_height)))
# rendering is torch tensor of shape (3, image_height, image_width), use interpolate.griddata to interpolate the accurate points
# first interpolate the red channel
red = rendering[0].cpu().detach().numpy().T
# accurate_red = np.array([red[x,y] for x,y in accurate_points])
# instead of for loop, we can use numpy fancy indexing
accurate_red = red[accurate_points[:,0], accurate_points[:,1]]
red_interpolated = interpolate.griddata(accurate_points, accurate_red, all_points, method='linear')
# then interpolate the green channel
green = rendering[1].cpu().detach().numpy().T
accurate_green = np.array([green[x,y] for x,y in accurate_points])
green_interpolated = interpolate.griddata(accurate_points, accurate_green, all_points, method='linear')
# then interpolate the blue channel
blue = rendering[2].cpu().detach().numpy().T
accurate_blue = np.array([blue[x,y] for x,y in accurate_points])
blue_interpolated = interpolate.griddata(accurate_points, accurate_blue, all_points, method='linear')
# combine the three channels
interpolated_rendering = torch.tensor([red_interpolated, green_interpolated, blue_interpolated], dtype=torch.float32, device="cuda")
# reshape to (3, image_width, image_height)
interpolated_rendering = interpolated_rendering.reshape(3, view.image_width, view.image_height)
# transpose the last two dimensions to get the usual convention
interpolated_rendering = interpolated_rendering.permute(0,2,1)

torchvision.utils.save_image(interpolated_rendering, "AMR_render_view%d.png"%camera_idx)
