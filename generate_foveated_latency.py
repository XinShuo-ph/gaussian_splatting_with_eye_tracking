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
import mpi4py

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


# pix_x = 1920
# pix_y = 1080

# same args as render.py
parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

parser.add_argument("--pix_x", default=1920, type=int)
parser.add_argument("--pix_y", default=1080, type=int)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
# parser.add_argument("--gaze_x", default=0, type=float)
# parser.add_argument("--gaze_y", default=0, type=float)
# parser.add_argument("--gaze_r2", default=1e4, type=float)
# parser.add_argument("--gaze_r3", default=1e4, type=float)
# parser.add_argument("--gaze_r4", default=1e4, type=float)
parser.add_argument("--percentile_r2", default=0.25, type=float)
parser.add_argument("--percentile_r3", default=0.5, type=float)
parser.add_argument("--percentile_r4", default=0.9, type=float)
parser.add_argument("--test_no_render_laststep", action="store_true") # test the time of purely passing the data in foveastep 4
parser.add_argument("--show_fps", action="store_true") # show fps, otherwise show latency
parser.add_argument("--interp", action="store_true") # whether do an interpolation to get the final image or just keep the blank pixels
parser.add_argument("--control_level_by_r", action="store_true") # control the foveation level by the radius
parser.add_argument("--debug_on", action="store_true") # debug mode



parser.add_argument("--foveal_model_path", default="/home/ubuntu/gaussian_splatting_with_eye_tracking/fovealnet/results_epoch/epoch_26/model_epoch_26.pt", type=str)
parser.add_argument("--eye_image_sequence_folder", default="/home/ubuntu/openeds/train/sequences/", type=str)
parser.add_argument("--eye_image_sequence_id_start", default=6400, type=int)
parser.add_argument("--eye_image_sequence_id_end", default=6499, type=int)
parser.add_argument("--foveal_output_file", type=str, default="predictions.txt")
parser.add_argument("--foveal_layer_timer", action="store_true")

args = get_combined_args(parser)
pix_x = args.pix_x
pix_y = args.pix_y
print("Rendering " + args.model_path)
safe_state(args.quiet)
mydataset = model.extract(args)

if args.debug_on:
    pipeline.debug = True
    print("Debug mode on")






# also define vison transformer here to use mpi communication

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
            # output in all 6 layers, instead of only 1,3,5
            # if i % 2 == 1 and self.score_method == "attention": 
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
        

# get scene name
model_path_to_scene = {
"output/e26eae8e-f": "playroom",
"output/29554d64-8": "drjohnson",
"output/06008696-3": "train",
"output/36cf0258-6": "truck"
}
scene_name = model_path_to_scene[args.model_path]
print("Scene: ", scene_name)

# render the acurate image for reference
gaussians = GaussianModel(mydataset.sh_degree) # create an empty gaussian model
scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False) # load the model and cameras
views = scene.getTrainCameras()
views = [views[i] for i in range(100)] # use 100 views to average the fps
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

pix_horizon = []
fps_avg = []

# runnames = ['foveated_GPU', 'foveated', 'A3FR_renderonly', 'A3FR_GPU_CPU', 'original','AMR']
runnames = ['original', 'foveated_GPU', 'A3FR_GPU_CPU', 'A3FR_renderonly', 'foveated', 'AMR']

# 'original': original rendering with full resolution
# 'AMR': adaptive multi-resolution rendering, but not foveation
# 'foveated': foveation rendering, but not adaptive multi-resolution
# 'foveated_GPU': foveation rendering, with a fovealnet running in serial on GPU
# 'A3FR_GPU_CPU': adaptive multi-resolution rendering and foveation rendering, with a fovealnet running in parallel on CPU
# 'A3FR_renderonly': adaptive multi-resolution rendering and foveation rendering, but not fovealnet running

fov = 120 # 120 degree field of view
r2_deg = 33 # 30 degree radius for peripheral vision
r3_deg = 26 # 9 degree radius for near fovea
r4_deg = 18 # 4 degree radius for fovea center
gaze_x=0 # define the var
gaze_y=0
gaze_r2=1e4
gaze_r3=1e4
gaze_r4=1e4

for ratio in [  2.0/3.0 ,  1  ,  4.0/3.0  ]:
# for ratio in [ 1 ]:

    print(f"Rendering at {int(pix_x*ratio)}x{int(pix_y*ratio)}")
    # change the image width and height
    for i in range(len(views)):
        views[i].image_width = int(pix_x*ratio)
        views[i].image_height = int(pix_y*ratio)
    
    # open file "latency_%dp.txt"%pix_y in append mode
    f = open("latency_%dp.txt"%int(pix_y*ratio), "a")
    f.write(scene_name + " ")


    for runidx, runname in enumerate(runnames):
        print(f"running {runname}...")

        if runname == 'foveated' or runname == 'A3FR_GPU_CPU' or runname == 'A3FR_renderonly' or runname == 'foveated_GPU':
            gaze_x = int(pix_x*ratio/2)
            gaze_y = int(pix_y*ratio/2)
            gaze_r2 = r2_deg/fov * pix_x*ratio
            gaze_r3 = r3_deg/fov * pix_x*ratio
            gaze_r4 = r4_deg/fov * pix_x*ratio
            print(f"gaze_x: {gaze_x}, gaze_y: {gaze_y}, gaze_r2: {gaze_r2}, gaze_r3: {gaze_r3}, gaze_r4: {gaze_r4}")

        if runname == 'A3FR_GPU_CPU':
            os.system("screen -S resnet_timing -X stuff \"while true; do python track.py --cpu_infer --layer_timer; done$(printf \\\\r)\"")

        if runname == 'foveated_GPU':
            device = torch.device("cuda" if (torch.cuda.is_available() ) else "cpu")
            fovmodel = VisionTransformer(num_layers=6, top_k=1.0).to(device)
            
            num_events = 8 # +1 for patch embedding, +1 for final layers
            fovstarters = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
            fovenders = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]

            sequence_folder = os.path.join(args.eye_image_sequence_folder, f"{args.eye_image_sequence_id_start:04d}")
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
                if runname == 'A3FR_GPU_CPU':
                    rendering = render(view, gaussians, pipeline, background,
                                    gaze_x=gaze_x, gaze_y=gaze_y, gaze_r2=gaze_r2, gaze_r3=gaze_r3, gaze_r4=gaze_r4,
                                    percentile_r2=args.percentile_r2, percentile_r3=args.percentile_r3, percentile_r4=args.percentile_r4,
                                starter = starter, ender= ender, 
                                starters = [starter0, starter1, starter2, starter3, starter4], enders = [ender0, ender1, ender2, ender3, ender4],
                                test_no_render_laststep = args.test_no_render_laststep, interpolate_image = args.interp, control_level_by_r = args.control_level_by_r
                                )["render"]
                elif runname == 'A3FR_renderonly':
                    rendering = render(view, gaussians, pipeline, background,
                                    gaze_x=gaze_x, gaze_y=gaze_y, gaze_r2=gaze_r2, gaze_r3=gaze_r3, gaze_r4=gaze_r4,
                                    percentile_r2=args.percentile_r2, percentile_r3=args.percentile_r3, percentile_r4=args.percentile_r4,
                                starter = starter, ender= ender, 
                                starters = [starter0, starter1, starter2, starter3, starter4], enders = [ender0, ender1, ender2, ender3, ender4],
                                test_no_render_laststep = args.test_no_render_laststep, interpolate_image = args.interp, control_level_by_r = args.control_level_by_r
                                )["render"]
                elif runname == 'original':
                    rendering = render(view, gaussians, pipeline, background,
                                    percentile_r2=0, percentile_r3=0, percentile_r4=0,
                                starter = starter, ender= ender, 
                                starters = [starter0, starter1, starter2, starter3, starter4], enders = [ender0, ender1, ender2, ender3, ender4],
                                test_no_render_laststep = args.test_no_render_laststep, interpolate_image = args.interp, control_level_by_r = args.control_level_by_r
                                )["render"]
                elif runname == 'AMR':
                    rendering = render(view, gaussians, pipeline, background,
                                    percentile_r2=args.percentile_r2, percentile_r3=args.percentile_r3, percentile_r4=args.percentile_r4,
                                starter = starter, ender= ender, 
                                starters = [starter0, starter1, starter2, starter3, starter4], enders = [ender0, ender1, ender2, ender3, ender4],
                                test_no_render_laststep = args.test_no_render_laststep, interpolate_image = args.interp, control_level_by_r = args.control_level_by_r
                                )["render"]
                elif runname == 'foveated':
                    rendering = render(view, gaussians, pipeline, background,
                                    gaze_x=gaze_x, gaze_y=gaze_y, gaze_r2=gaze_r2, gaze_r3=gaze_r3, gaze_r4=gaze_r4,
                                    percentile_r2=0, percentile_r3=0, percentile_r4=0,
                                starter = starter, ender= ender, 
                                starters = [starter0, starter1, starter2, starter3, starter4], enders = [ender0, ender1, ender2, ender3, ender4],
                                test_no_render_laststep = args.test_no_render_laststep, interpolate_image = args.interp, control_level_by_r = args.control_level_by_r
                                )["render"]
                elif runname == 'foveated_GPU':
                    # run fovealnet on GPU
                    
                    image_name = f"{idx%50:03d}.png"
                    image_path = os.path.join(sequence_folder, image_name)
                    image = load_image(image_path).to(device)
                    _ = fovmodel.forward_timer(image, fovstarters, fovenders)
                    rendering = render(view, gaussians, pipeline, background,
                                    gaze_x=gaze_x, gaze_y=gaze_y, gaze_r2=gaze_r2, gaze_r3=gaze_r3, gaze_r4=gaze_r4,
                                    percentile_r2=0, percentile_r3=0, percentile_r4=0,
                                starter = starter, ender= ender, 
                                starters = [starter0, starter1, starter2, starter3, starter4], enders = [ender0, ender1, ender2, ender3, ender4],
                                test_no_render_laststep = args.test_no_render_laststep, interpolate_image = args.interp, control_level_by_r = args.control_level_by_r
                                )["render"]
                torch.cuda.synchronize()
                time += starter.elapsed_time(ender)
                if runname == 'foveated_GPU':
                    for timeridx in range(num_events):
                        if timeridx == num_events-1:
                            continue
                        time += fovstarters[timeridx].elapsed_time(fovenders[timeridx])
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
        
        if runname == 'A3FR_GPU_CPU':
            # command = "screen -S resnet_timing -X stuff \"\^C\""
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")
            os.system("screen -S resnet_timing -X stuff \"^C\"")

        # test = 1/(1/avg_fps0 + 1/avg_fps1 + 1/avg_fps2 + 1/avg_fps3 + 1/avg_fps4)

        # pix_horizon.append(int(pix_x*ratio))
        # fps_avg.append(avg_fps)
        f.write(f"{avg_fps} ")
    f.write("\n")
    f.close()


# torchvision.utils.save_image(rendering, "tmp4.png")