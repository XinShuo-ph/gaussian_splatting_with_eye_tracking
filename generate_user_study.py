import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_amr import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer_amr import GaussianModel
import time
from scipy import interpolate
from itertools import product

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)


parser.add_argument("--pix_x", default=1920, type=int)
parser.add_argument("--pix_y", default=1080, type=int)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--gaze_x", default=0, type=float)
parser.add_argument("--gaze_y", default=0, type=float)
parser.add_argument("--gaze_r2", default=1e4, type=float)
parser.add_argument("--gaze_r3", default=1e4, type=float)
parser.add_argument("--gaze_r4", default=1e4, type=float)
parser.add_argument("--percentile_r2", default=0.25, type=float)
parser.add_argument("--percentile_r3", default=0.5, type=float)
parser.add_argument("--percentile_r4", default=0.9, type=float)
parser.add_argument("--test_no_render_laststep", action="store_true") # test the time of purely passing the data in foveastep 4
parser.add_argument("--interp", action="store_true") # whether do an interpolation to get the final image or just keep the blank pixels
parser.add_argument("--control_level_by_r", action="store_true") # control the foveation level by the radius
args = get_combined_args(parser)
pix_x = args.pix_x
pix_y = args.pix_y
print("Rendering " + args.model_path)
safe_state(args.quiet)
mydataset = model.extract(args)

gaussians = GaussianModel(mydataset.sh_degree) # create an empty gaussian model
scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False) # load the model and cameras
views = scene.getTrainCameras()
views = [views[i] for i in range(100)] # use 100 views to average the fps
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

# get scene name
model_path_to_scene = {
"output/e26eae8e-f": "playroom",
"output/29554d64-8": "drjohnson",
"output/06008696-3": "train",
"output/36cf0258-6": "truck"
}
scene_name = model_path_to_scene[args.model_path]
print("Scene: ", scene_name)


for idx in range(5):

    # skip is "user_study/original_marked/"+scene_name+"%d.png"%idx exists
    if os.path.exists("user_study/original_marked/"+scene_name+"%d.png"%idx):
        print("Skip ", idx)
        continue

    viewidx = 21*idx

    views[viewidx].image_width = int(pix_x)
    views[viewidx].image_height = int(pix_y)
    view = views[viewidx]


    # accurate rendering
    original_rendering = render(view, gaussians, pipeline, background,
                                    percentile_r2=0, percentile_r3=0, percentile_r4=0
                                )["render"]

    torchvision.utils.save_image(original_rendering, "user_study/original/"+scene_name+"%d.png"%idx)

    
    # mark the gaze point
    original_rendering_marked = original_rendering.clone()
    # Define the size of the cross
    cross_size = int(view.image_width/50)  # Adjust the size as needed
    # Convert gaze coordinates to integer indices
    gaze_x_idx = int(args.gaze_x)
    gaze_y_idx = int(args.gaze_y)
    # Ensure the cross doesn't go out of bounds
    x_start = max(gaze_x_idx - cross_size, 0)
    x_end = min(gaze_x_idx + cross_size+1, view.image_width)
    y_start = max(gaze_y_idx - cross_size, 0)
    y_end = min(gaze_y_idx + cross_size+1, view.image_height)
    # Draw the horizontal line of the cross
    original_rendering_marked[0, y_start:y_end, gaze_x_idx] = 1.0  # Red channel
    original_rendering_marked[1, y_start:y_end, gaze_x_idx] = 0.0  # Green channel
    original_rendering_marked[2, y_start:y_end, gaze_x_idx] = 0.0  # Blue channel
    # Draw the vertical line of the cross
    original_rendering_marked[0, gaze_y_idx, x_start:x_end] = 1.0  # Red channel
    original_rendering_marked[1, gaze_y_idx, x_start:x_end] = 0.0  # Green channel
    original_rendering_marked[2, gaze_y_idx, x_start:x_end] = 0.0  # Blue channel

    # Save the image with the red cross
    torchvision.utils.save_image(original_rendering_marked, "user_study/original_marked/"+scene_name+"%d.png"%idx)


    rendering = render(view, gaussians, pipeline, background,
                                gaze_x=args.gaze_x, gaze_y=args.gaze_y, gaze_r2=args.gaze_r2, gaze_r3=args.gaze_r3, gaze_r4=args.gaze_r4,
                                    percentile_r2=args.percentile_r2, percentile_r3=args.percentile_r3, percentile_r4=args.percentile_r4,
                                test_no_render_laststep = args.test_no_render_laststep, interpolate_image = args.interp, control_level_by_r = args.control_level_by_r
                                )["render"]

    torchvision.utils.save_image(rendering, "user_study/foveated_uninterpolated/"+scene_name+"%d.png"%idx)
    
    # interpolate the rendering
    red = rendering[0].cpu().detach().numpy().T
    green = rendering[1].cpu().detach().numpy().T
    blue = rendering[2].cpu().detach().numpy().T
    accurate_points = np.array(np.where(red>0)).T
    missed_points = np.array(np.where(red==0)).T
    all_points = list(product(range(view.image_width), range(view.image_height)))
    accurate_red = red[accurate_points[:,0], accurate_points[:,1]]
    start = time.time()
    red_interpolated = interpolate.griddata(accurate_points, accurate_red, all_points, method='linear')
    accurate_green = np.array([green[x,y] for x,y in accurate_points])
    green_interpolated = interpolate.griddata(accurate_points, accurate_green, all_points, method='linear')
    accurate_blue = np.array([blue[x,y] for x,y in accurate_points])
    blue_interpolated = interpolate.griddata(accurate_points, accurate_blue, all_points, method='linear')
    end=time.time()
    print("Time for interpolation: ", end-start)
    interpolated_rendering = torch.tensor([red_interpolated, green_interpolated, blue_interpolated], dtype=torch.float32, device="cuda")
    # reshape to (3, image_width, image_height)
    interpolated_rendering = interpolated_rendering.reshape(3, view.image_width, view.image_height)
    # transpose the last two dimensions to get the usual convention
    interpolated_rendering = interpolated_rendering.permute(0,2,1)

    torchvision.utils.save_image(interpolated_rendering, "user_study/foveated/"+scene_name+"%d.png"%idx)

    # mark the gaze point
    interpolated_rendering_marked = interpolated_rendering.clone()
    # Define the size of the cross
    cross_size = int(view.image_width/50)  # Adjust the size as needed
    # Convert gaze coordinates to integer indices
    gaze_x_idx = int(args.gaze_x)
    gaze_y_idx = int(args.gaze_y)
    # Ensure the cross doesn't go out of bounds
    x_start = max(gaze_x_idx - cross_size, 0)
    x_end = min(gaze_x_idx + cross_size+1, view.image_width)
    y_start = max(gaze_y_idx - cross_size, 0)
    y_end = min(gaze_y_idx + cross_size+1, view.image_height)
    # Draw the horizontal line of the cross
    interpolated_rendering_marked[0, y_start:y_end, gaze_x_idx] = 1.0  # Red channel
    interpolated_rendering_marked[1, y_start:y_end, gaze_x_idx] = 0.0  # Green channel
    interpolated_rendering_marked[2, y_start:y_end, gaze_x_idx] = 0.0  # Blue channel
    # Draw the vertical line of the cross
    interpolated_rendering_marked[0, gaze_y_idx, x_start:x_end] = 1.0  # Red channel
    interpolated_rendering_marked[1, gaze_y_idx, x_start:x_end] = 0.0  # Green channel
    interpolated_rendering_marked[2, gaze_y_idx, x_start:x_end] = 0.0  # Blue channel

    # Save the image with the red cross
    torchvision.utils.save_image(interpolated_rendering_marked, "user_study/foveated_marked/"+scene_name+"%d.png"%idx)

