import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_amr import render_once
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer_amr import GaussianModel

pix_x = 1920
pix_y = 1080

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
views = scene.getTrainCameras()
views = [views[i] for i in range(100)] # use 100 views to average the fps
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

pix_horizon = []
fps_avg = []
for ratio in [1]:

    print(f"Rendering at {int(pix_x*ratio)}x{int(pix_y*ratio)}")
    # change the image width and height
    for i in range(len(views)):
        views[i].image_width = int(pix_x*ratio)
        views[i].image_height = int(pix_y*ratio)


    # test fps by counting time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    fpss = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        time = 0
        for i in range(5):
            rendering = render_once(view, gaussians, pipeline, background,starter = starter, ender= ender)["render"]
            torch.cuda.synchronize()
            time += starter.elapsed_time(ender)
        # count fps every 5 frames
        fps = 5 / (time / 1000)
        # print("FPS: ", fps)
        fpss.append(fps)

        
    avg_fps = sum(fpss) / len(fpss)
    print(f"Average FPS: {avg_fps}")
    pix_horizon.append(int(pix_x*ratio))
    fps_avg.append(avg_fps)
