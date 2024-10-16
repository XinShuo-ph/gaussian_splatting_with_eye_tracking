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

# same args as render.py
parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

# pipeline.debug = True

parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
args = get_combined_args(parser)
print("Rendering " + args.model_path)
safe_state(args.quiet)
mydataset = model.extract(args)


gaussians = GaussianModel(mydataset.sh_degree) # create an empty gaussian model
scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False) # load the model and cameras
views = scene.getTrainCameras()
views = [views[i] for i in range(100)] # use 100 views to average the fps
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")



pix_horizon = []
fps_avg = []
fps_avg0 = []
fps_avg1 = []
fps_avg2 = []
fps_avg3 = []
for ratio in np.linspace(0.2,2.0,10):

    # restart cuda kernel and reload the model to avoid memory leak
    # torch.cuda.reset_max_memory_allocated()
    # torch.cuda.reset_peak_memory_stats()
    
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # if 'gaussians' in locals():
    #     del gaussians
    # if 'scene' in locals():
    #     del scene
    # if 'views' in locals():
    #     del views
    # if 'background' in locals():
    #     del background
    # if 'result' in locals():
    #     del result
    # if 'rendering' in locals():
    #     del rendering


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

    torch.cuda.synchronize()
    fpss = []
    fpss0 = []
    fpss1 = []
    fpss2 = []
    fpss3 = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        time = 0
        time0 = 0
        time1 = 0
        time2 = 0
        time3 = 0
        for i in range(5):
            result = render(view, gaussians, pipeline, background,starter = starter, ender= ender, 
                               starters = [starter0, starter1, starter2, starter3], enders = [ender0, ender1, ender2, ender3]
            )
            rendering = result["render"]
            torch.cuda.synchronize()
            time += starter.elapsed_time(ender)
            time0 += starter0.elapsed_time(ender0)
            time1 += starter1.elapsed_time(ender1)
            time2 += starter2.elapsed_time(ender2)
            time3 += starter3.elapsed_time(ender3)
        # count fps every 5 frames
        fps = 5 / (time / 1000)
        fps0 = 5 / (time0 / 1000)
        fps1 = 5 / (time1 / 1000)
        fps2 = 5 / (time2 / 1000)
        fps3 = 5 / (time3 / 1000)
        # print("FPS: ", fps)
        fpss.append(fps)
        fpss0.append(fps0)
        fpss1.append(fps1)
        fpss2.append(fps2)
        fpss3.append(fps3)

        
    avg_fps = sum(fpss) / len(fpss)
    avg_fps0 = sum(fpss0) / len(fpss0)
    avg_fps1 = sum(fpss1) / len(fpss1)
    avg_fps2 = sum(fpss2) / len(fpss2)
    avg_fps3 = sum(fpss3) / len(fpss3)

    print(f"Average FPS: {avg_fps}")
    print(f"Average FPS of fov level 0: {avg_fps0}")
    print(f"Average FPS of fov level 1: {avg_fps1}")
    print(f"Average FPS of fov level 2: {avg_fps2}")
    print(f"Average FPS of fov level 3: {avg_fps3}")

    pix_horizon.append(int(pix_x*ratio))
    fps_avg.append(avg_fps)
    fps_avg0.append(avg_fps0)
    fps_avg1.append(avg_fps1)
    fps_avg2.append(avg_fps2)
    fps_avg3.append(avg_fps3)


    
# plot the fps vs horizontal pix curve for all steps
import matplotlib.pyplot as plt
plt.plot(pix_horizon, fps_avg, 'ko-',label = 'total')
plt.plot(pix_horizon, fps_avg0, 'ro-',label = 'preprocess + level 1')
plt.plot(pix_horizon, fps_avg1, 'go-',label = 'level 2')
plt.plot(pix_horizon, fps_avg2, 'bo-',label = 'level 3')
plt.plot(pix_horizon, fps_avg3, 'yo-',label = 'level 4')
plt.xlabel('Horizontal Pixels')
plt.ylabel('FPS')
plt.title('model: 3DGS_AMR, machine: T4')
plt.legend()
plt.savefig('fps_benchmark_amr_fovsteps.png')

plt.figure()
plt.plot(pix_horizon, fps_avg, 'ko-',label = 'total')
plt.plot(pix_horizon, fps_avg0, 'ro-',label = 'preprocess + level 1')
plt.plot(pix_horizon, fps_avg1, 'go-',label = 'level 2')
plt.plot(pix_horizon, fps_avg2, 'bo-',label = 'level 3')
plt.plot(pix_horizon, fps_avg3, 'yo-',label = 'level 4')
plt.plot(np.linspace(800,4000,100), 1.5e8/np.linspace(800,4000,100)**2, 'k--',label = '$y\\propto x^{-2}$')
plt.plot(np.linspace(800,4000,100), 7e4/np.linspace(800,4000,100), 'b--',label = '$y\\propto x^{-1}$')
plt.xlabel('Horizontal Pixels')
plt.ylabel('FPS')
plt.title('model: 3DGS_AMR, machine: T4')
plt.legend()
plt.ylim(1,120)
plt.savefig('fps_benchmark_scale_amr_fovsteps.png')


plt.figure()
plt.plot(pix_horizon, fps_avg, 'ko-',label = 'total')
plt.plot(pix_horizon, fps_avg0, 'ro-',label = 'preprocess + level 1')
plt.plot(pix_horizon, fps_avg1, 'go-',label = 'level 2')
plt.plot(pix_horizon, fps_avg2, 'bo-',label = 'level 3')
plt.plot(pix_horizon, fps_avg3, 'yo-',label = 'level 4')
plt.plot(np.linspace(800,4000,100), 1.8e8/np.linspace(800,4000,100)**2, 'k--',label = '$y\\propto x^{-2}$')
plt.plot(np.linspace(800,4000,100), 7e4/np.linspace(800,4000,100), 'b--',label = '$y\\propto x^{-1}$')
plt.semilogy(np.linspace(800,4000,100), 140*np.exp( - 0.75* np.linspace(800,4000,100)/1000), 'y--',label = '$\\log y \\propto C - x $')
plt.xlabel('Horizontal Pixels')
plt.ylabel('FPS')
plt.title('model: 3DGS_AMR, machine: T4')
plt.legend()
plt.ylim(10,120)
plt.savefig('fps_benchmark_log_amr_fovsteps.png')

