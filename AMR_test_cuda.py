import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import product


# test the output from cuda rasterization
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


pix_x = 1920
pix_y = 1080
camera_idx = 0
tile_level = 5 # 2^tile_level is the number of pixels on a tile edge
AMR_factor = 0.5
AMR_MAX_LEVELS_count = 4
AMR_level_percentiles = [0.1, 0.4, 0.9,1.0]


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



pipeline.debug = True


if pipeline.debug:
    from gaussian_renderer_amr_debug import render
    from gaussian_renderer_amr_debug import GaussianModel
else:
    from gaussian_renderer_amr import render
    from gaussian_renderer_amr import GaussianModel

# render the acurate image for reference
gaussians = GaussianModel(mydataset.sh_degree) # create an empty gaussian model
scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False) # load the model and cameras


view = scene.getTrainCameras()[camera_idx] # get the camera
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

view.image_width = int(pix_x)
view.image_height = int(pix_y)

result = render(view, gaussians, pipeline, background)
rendering = result["render"]
gaussians_2d_r = result["radii"].cpu().detach().numpy()
torchvision.utils.save_image(rendering, "tmp.png")
# count the number of skipped pixels
reds = rendering[0].cpu().detach().numpy()
redmask = reds == 0
print("Skipped pixels: ", np.sum(redmask))
print("Total pixels: ", redmask.size)

# project the gaussians to 2d screen and plot the circles

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
                points_view[:,2] > 0.2) &(
                    gaussians_2d_r>0
                ) # only keep the points in front of the camera, refer to gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h:154,
gaussians_2d_x = gaussians_2d_x[mask]
gaussians_2d_y = gaussians_2d_y[mask]
gaussians_2d_r = gaussians_2d_r[mask]


plt.figure().set_size_inches(10,6)
plt.imshow(rendering.cpu().detach().numpy().transpose(1,2,0))
# put the mesh on top of the image
plt.axhline(0, color='r', linewidth=3, alpha=0.5, label="tiles")
for x in range(0,view.image_height,2**tile_level):
    plt.axhline(x, color='r', linewidth=0.5, alpha=0.5)


for y in range(0,view.image_width,2**tile_level):
    plt.axvline(y, color='r', linewidth=0.5, alpha=0.5)

# plot all the gaussians by centered on x,y, with radius r
# fill them with blue, alpha=0.1
# Turn off interactive mode and suppress output
# plt.ioff()
# for x,y,r in zip(gaussians_2d_x, gaussians_2d_y, gaussians_2d_r):
#     circle = plt.Circle((x, y), r, color='b', alpha=0.1)
#     plt.gca().add_artist(circle)


# plt.draw()

# above is too slow, plot by plt.scatter and control the size of the points by gaussians_2d_r
# plt.scatter(gaussians_2d_x, gaussians_2d_y, s=gaussians_2d_r**2, c='b', alpha=0.01)


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

# scatter the gaussians
plt.scatter(gaussians_2d_x[0], gaussians_2d_y[0], marker='.', color='b', s=20, alpha=1, label="gaussians") 
plt.scatter(gaussians_2d_x, gaussians_2d_y, marker='.', color='b', s=0.05, alpha=0.2)


# note: this below does not turn out to be good vis, entire image is covered by the circles

# # Get the current figure size in inches and data limits
# fig_width_in, fig_height_in = (10,6)
# x_min, x_max = (-1, view.image_width)
# y_min, y_max = (-1, view.image_height)

# # Calculate the number of data units per point
# x_points_per_unit = fig_width_in * 72 / (x_max - x_min)
# y_points_per_unit = fig_height_in * 72 / (y_max - y_min)

# # Use the average as our conversion factor
# points_per_unit = (x_points_per_unit + y_points_per_unit) / 2

# plt.scatter(gaussians_2d_x, gaussians_2d_y, s=(gaussians_2d_r*points_per_unit)**2, c='b', alpha=0.005)


plt.savefig("amr_render_all_gaussians_view%d.png"%camera_idx)


tiles_num_x = (view.image_width + 2**tile_level -1)// 2**tile_level  # from 0 to view.image_width//2**tile_level
tiles_num_y = (view.image_height + 2**tile_level -1) // 2**tile_level 

# record the ranges for each tile, i.e. the number of intersections and plot them here, to try a better AMR criterion
means2D = result["means2D"].cpu().detach().numpy()
ranges = result["ranges"].cpu().detach().numpy()

gaussians_2d_x = means2D[:,0]
gaussians_2d_y = means2D[:,1]
tiles_gaussian_count = ranges[:,1] - ranges[:,0]
tiles_gaussian_count = np.reshape(tiles_gaussian_count, (tiles_num_y, tiles_num_x)).T


tile_count_image = np.zeros((view.image_height, view.image_width)) 

for x in range(tiles_num_x):
    for y in range(tiles_num_y):
        start_x = x * 2**tile_level
        end_x = min((x + 1) * 2**tile_level, view.image_width)
        start_y = y * 2**tile_level
        end_y = min((y + 1) * 2**tile_level, view.image_height)
        tile_count_image[start_y:end_y, start_x:end_x] = tiles_gaussian_count[x, y]


# # set levels by log
# tiles_AMRlevel = np.floor( AMR_factor* np.log10(tiles_gaussian_count+1)* AMR_MAX_LEVELS_count/4).astype(int) +1 # a simple criterion, using log10(count) to determine the AMR level
# tiles_AMRlevel_image = np.floor(AMR_factor * np.log10(tile_count_image+1)* AMR_MAX_LEVELS_count/4).astype(int) +1

# tiles_AMRlevel[tiles_AMRlevel > 4] = 4
# tiles_AMRlevel_image[tiles_AMRlevel_image > 4] = 4


# # set levels linearly
# tiles_AMRlevel_image = np.floor( tile_count_image/700 ).astype(int) +1
# tiles_AMRlevel_image[tiles_AMRlevel_image > 4] = 4

# set levels by percentile
tiles_AMRlevel = np.zeros((tiles_num_x, tiles_num_y))
tiles_AMRlevel_image = np.zeros((view.image_height, view.image_width))

for i in range(AMR_MAX_LEVELS_count):
    percentile = AMR_level_percentiles[i]
    threshold_up = np.percentile(tiles_gaussian_count, percentile*100)
    threshold_down = np.percentile(tiles_gaussian_count, AMR_level_percentiles[i-1]*100) if i > 0 else 0
    tiles_AMRlevel[(tiles_gaussian_count > threshold_down) & (tiles_gaussian_count <= threshold_up)] = i+1
    tiles_AMRlevel_image[(tile_count_image > threshold_down) & (tile_count_image <= threshold_up)] = i+1


# now imshow the tiles_gaussian_count, but note that each pix for tiles_gaussian_count should cover 2^tile_level pixels in previous image
plt.figure().set_size_inches(10,6)
plt.imshow(rendering.cpu().detach().numpy().transpose(1,2,0))

# Overlay the tile count image with transparency and log scale
# plt.imshow(np.log10(tile_count_image), cmap='jet',  alpha=0.5)
# plt.colorbar(label='$\\log_{10}$ # of intersections')


plt.imshow((tile_count_image), cmap='jet',  alpha=0.5,vmin =0,vmax = 2000)
plt.colorbar(label=' # of intersections')

plt.xlabel("horizontal pixel", fontsize=16)
plt.ylabel("vertical pixel", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(-1,view.image_width)
plt.ylim(-1,view.image_height)
plt.gca().invert_yaxis()
plt.savefig("amr_render_tile_intersections_view%d.png"%camera_idx)




plt.figure().set_size_inches(10,6)
plt.imshow(rendering.cpu().detach().numpy().transpose(1,2,0))

# Overlay the tile count image with AMR level, note that the level are discrete numbers 1,2,3,4, adjust the cmap accordingly

# Define discrete colormap
cmap = ListedColormap(['blue', 'cyan', 'yellow', 'red'])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
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
plt.savefig("amr_render_tile_AMR_view%d.png"%camera_idx)


# cross check with the AMR_levels output by cuda code

tiles_AMRlevel_cuda = result["tile_AMR_levels"].cpu().detach().numpy()
tiles_AMRlevel_image_cuda = np.zeros((view.image_height, view.image_width)) 

for x in range(tiles_num_x):
    for y in range(tiles_num_y):
        start_x = x * 2**tile_level
        end_x = min((x + 1) * 2**tile_level, view.image_width)
        start_y = y * 2**tile_level
        end_y = min((y + 1) * 2**tile_level, view.image_height)
        tiles_AMRlevel_image_cuda[start_y:end_y, start_x:end_x] = tiles_AMRlevel_cuda[x+ y * tiles_num_x ]




plt.figure().set_size_inches(10,6)
plt.imshow(rendering.cpu().detach().numpy().transpose(1,2,0))

# Overlay the tile count image with AMR level, note that the level are discrete numbers 1,2,3,4, adjust the cmap accordingly

# Define discrete colormap
cmap = ListedColormap(['blue', 'cyan', 'yellow', 'red'])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = BoundaryNorm(bounds, cmap.N)
plt.imshow(tiles_AMRlevel_image_cuda, cmap=cmap, norm=norm, alpha=0.5)

# Add colorbar to show the scale of AMR levels
plt.colorbar(label='AMR level', ticks=[1, 2, 3, 4])

plt.xlabel("horizontal pixel", fontsize=16)
plt.ylabel("vertical pixel", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(-1,view.image_width)
plt.ylim(-1,view.image_height)
plt.gca().invert_yaxis()
plt.savefig("amr_render_tile_AMR_cuda_view%d.png"%camera_idx)

