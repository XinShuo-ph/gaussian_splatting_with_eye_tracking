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

# pix_x = 1920
# pix_y = 1080

# same args as render.py
parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

print(pipeline.debug)

# pipeline.debug = True

print(pipeline.debug)

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
parser.add_argument("--show_fps", action="store_true") # show fps, otherwise show latency
parser.add_argument("--interp", action="store_true") # whether do an interpolation to get the final image or just keep the blank pixels
parser.add_argument("--control_level_by_r", action="store_true") # control the foveation level by the radius
args = get_combined_args(parser)
pix_x = args.pix_x
pix_y = args.pix_y
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

    # record 4 fov steps separately
    starter0, ender0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter2, ender2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter3, ender3 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter4, ender4 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    fpss = []
    fpss0 = []
    fpss1 = []
    fpss2 = []
    fpss3 = []
    fpss4 = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        time = 0
        time0 = 0
        time1 = 0
        time2 = 0
        time3 = 0
        time4 = 0
        for i in range(5):
            rendering = render(view, gaussians, pipeline, background,
                               gaze_x=args.gaze_x, gaze_y=args.gaze_y, gaze_r2=args.gaze_r2, gaze_r3=args.gaze_r3, gaze_r4=args.gaze_r4,
                                 percentile_r2=args.percentile_r2, percentile_r3=args.percentile_r3, percentile_r4=args.percentile_r4,
                               starter = starter, ender= ender, 
                               starters = [starter0, starter1, starter2, starter3, starter4], enders = [ender0, ender1, ender2, ender3, ender4],
                               test_no_render_laststep = args.test_no_render_laststep, interpolate_image = args.interp, control_level_by_r = args.control_level_by_r
                               )["render"]
            torch.cuda.synchronize()
            time += starter.elapsed_time(ender)
            time0 += starter0.elapsed_time(ender0)
            time1 += starter1.elapsed_time(ender1)
            time2 += starter2.elapsed_time(ender2)
            time3 += starter3.elapsed_time(ender3)
            time4 += starter4.elapsed_time(ender4)
        # count fps every 5 frames
        if args.show_fps:
            fps = 5 / (time / 1000)
            fps0 = 5 / (time0 / 1000)
            fps1 = 5 / (time1 / 1000)
            fps2 = 5 / (time2 / 1000)
            fps3 = 5 / (time3 / 1000)
            fps4 = 5 / (time4 / 1000)
        else: # otherwise pass latency in ms
            fps = time / 5
            fps0 = time0 / 5
            fps1 = time1 / 5
            fps2 = time2 / 5
            fps3 = time3 / 5
            fps4 = time4 / 5
        # print("FPS: ", fps)
        fpss.append(fps)
        fpss0.append(fps0)
        fpss1.append(fps1)
        fpss2.append(fps2)
        fpss3.append(fps3)
        fpss4.append(fps4)

        
    avg_fps = sum(fpss) / len(fpss)
    avg_fps0 = sum(fpss0) / len(fpss0)
    avg_fps1 = sum(fpss1) / len(fpss1)
    avg_fps2 = sum(fpss2) / len(fpss2)
    avg_fps3 = sum(fpss3) / len(fpss3)
    avg_fps4 = sum(fpss4) / len(fpss4)

    if args.show_fps:      
        print(f"Average FPS: {avg_fps}")
        print(f"Average FPS of fov level 0: {avg_fps0}")
        print(f"Average FPS of fov level 1: {avg_fps1}")
        print(f"Average FPS of fov level 2: {avg_fps2}")
        print(f"Average FPS of fov level 3: {avg_fps3}")
        print(f"Average FPS of fov level 4: {avg_fps4}")
    else:
        print(f"Average latency: {avg_fps} ms (including var passing in python)")
        print(f"Average latency of fov level 0: {avg_fps0} ms")
        print(f"Average latency of fov level 1: {avg_fps1} ms")
        print(f"Average latency of fov level 2: {avg_fps2} ms")
        print(f"Average latency of fov level 3: {avg_fps3} ms")
        print(f"Average latency of fov level 4: {avg_fps4} ms")
        

    # test = 1/(1/avg_fps0 + 1/avg_fps1 + 1/avg_fps2 + 1/avg_fps3 + 1/avg_fps4)

    pix_horizon.append(int(pix_x*ratio))
    fps_avg.append(avg_fps)


# torchvision.utils.save_image(rendering, "tmp.png")