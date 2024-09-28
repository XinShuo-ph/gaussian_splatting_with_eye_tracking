# opacity is known to be within 0-1, and contribute linearly to the final color
# therefore we can quantize the opacity to 2^n levels and save as n bit integer
# and save the residual opacity as a float
# then at lower resolution, we can use the integer to get a rough estimate
# at higher resolution, we can use the float (or higher bit integer) to correct the rough estimate

import numpy as np
from utils.general_utils import inverse_sigmoid

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

quant_bit = 1 # bit to represent opacity, 
# 1 bit effectively makes opacity 0 or 1 depending on > or < 0.5
# trials show 1 bit is pretty good already
my_eps = 1e-6 # avoid strict 1 or 0
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
gaussians = GaussianModel(mydataset.sh_degree) # create an empty gaussian model
scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False) # load the model and cameras

opacity_activation = gaussians.get_opacity - my_eps # note there's a sigmoid activation, ._opacity is the input to the activation, get_opacity is the output
# in cuda implementation, we should do the calculation below in a designed way
quant_opacity_activation = (torch.floor(opacity_activation * 2**quant_bit) +my_eps )/ (2**quant_bit - 1.0 + my_eps*2)
residual_opacity_activation = opacity_activation * 2**quant_bit - quant_opacity_activation*(2**quant_bit - 1) 
# this makes sure quant_opacity_activation and residual_opacity_activation are within 0-1, with
# opacity_activation = quant_opacity_activation * (2**quant_bit - 1)/2**quant_bit + residual_opacity_activation/2**quant_bit

# render quantized opacity and compare with correction
view = scene.getTrainCameras()[camera_idx]
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
rendering = render(view, gaussians, pipeline, background)["render"]
torchvision.utils.save_image(rendering, "original_render.png")

gaussians._opacity = inverse_sigmoid(quant_opacity_activation)
rendering = render(view, gaussians, pipeline, background)["render"]
torchvision.utils.save_image(rendering, "opacity_quant_render.png")

gaussians._opacity = inverse_sigmoid(residual_opacity_activation)
rendering_res = render(view, gaussians, pipeline, background)["render"]
torchvision.utils.save_image(rendering_res/2**quant_bit + rendering * (2**quant_bit - 1)/2**quant_bit, "opacity_correction_render.png")

gt = view.original_image[0:3, :, :]
torchvision.utils.save_image(gt,  "gt.png")

