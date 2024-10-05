# test the output from cuda rasterization
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


pix_x = 1920
pix_y = 1080
camera_idx = 0

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
view = scene.getTrainCameras()[camera_idx] # get the camera
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

view.image_width = int(pix_x)
view.image_height = int(pix_y)

rendering = render(view, gaussians, pipeline, background)["render"]

# count the number of skipped pixels
reds = rendering[0].cpu().detach().numpy()
mask = reds == 0
print("Skipped pixels: ", np.sum(mask))
print("Total pixels: ", mask.size)

# record the ranges for each tile, i.e. the number of intersections and plot them here, to try a better AMR criterion
