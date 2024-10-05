# the pruning process described in Sec. 3.2, 3.2 of 2407.00435 requires additional training
# would a simple filter by distance be sufficient?
# to efficiently filter out the points that are too far away
# use KD tree query_ball_point
from scipy.spatial import cKDTree
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


# same args as render.py
parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
args = get_combined_args(parser)

prune_distance = 7.0 # prune points that are further than this distance from the camera center
camera_idx = 0 # idx in all cameras


print("Rendering " + args.model_path)

safe_state(args.quiet)

mydataset = model.extract(args)
gaussians = GaussianModel(mydataset.sh_degree) # create an empty gaussian model
scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False) # load the model and cameras

# create a KD tree
pts = gaussians._xyz.cpu().detach().numpy()
mytree = cKDTree(pts)


# prune points by distance from a camera center
view = scene.getTrainCameras()[0]
mycenter = view.camera_center.cpu().detach().numpy()

subsetidxs = mytree.query_ball_point(mycenter, prune_distance)

print("using %d points out of %d" % (len(subsetidxs), len(pts)))
gaussians._xyz = gaussians._xyz[subsetidxs]
gaussians._opacity = gaussians._opacity[subsetidxs]
gaussians._features_dc = gaussians._features_dc[subsetidxs]
gaussians._features_rest = gaussians._features_rest[subsetidxs]
gaussians._rotation = gaussians._rotation[subsetidxs]
gaussians._scaling = gaussians._scaling[subsetidxs]


# check rendered image with gt
bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

rendering = render(view, gaussians, pipeline, background)["render"]
gt = view.original_image[0:3, :, :]
torchvision.utils.save_image(rendering, "KDtree_render.png")
torchvision.utils.save_image(gt,  "gt.png")



# scatter plot pts in 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:,0], pts[:,1], pts[:,2], 'bx')
ax.scatter(pts[:,0], pts[:,1], pts[:,2], 'r.',s=0.01)
plt.savefig("KDtree_scatter.png")

