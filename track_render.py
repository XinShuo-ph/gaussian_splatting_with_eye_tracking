# a demo code to render 3d gaussian splatting from a camera matrix defined by eyetracking result

# there should be 3 steps to render the splatting:
# 1. from the input eye image, segment the pupil (label=3), iris (label=2), sclera (label=1) and background (label=0)
#     - the segmentation model used is RITnet: https://bitbucket.org/eye-ush/ritnet/src/master/ https://arxiv.org/pdf/1910.00694 
# 2. from the segmentation result, estimate gaze direction and camera matrix
#     - the gaze estimation used is hmd-eyes: https://github.com/pupil-labs/hmd-eyes https://arxiv.org/pdf/2403.19768v1
# 3. render the 3d gaussian splatting from the camera matrix

import torch
from RITnet.dataset import IrisDataset
from torch.utils.data import DataLoader 
from PIL import Image
from torchvision import transforms
import cv2
import random
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from RITnet.dataset import transform
import os
from RITnet.opt import parse_args
from RITnet.models import model_dict
from tqdm import tqdm
from RITnet.utils import get_predictions
from render import *

# 0. arguments

parser = ArgumentParser(description="Testing script parameters")
splat_model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
args = get_combined_args(parser)
print("Rendering " + args.model_path)
# TODO: later we should parse the eye tracking parameters from command line
eye_model_type = "densenet"
eye_model_path = "RITnet/best_model.pkl"
eye_image_path = "eye.png"
useGPU = True 


# 1. from the input eye image, segment the pupil (label=3), iris (label=2), sclera (label=1) and background (label=0)
#     - the segmentation model used is RITnet: https://bitbucket.org/eye-ush/ritnet/src/master/ https://arxiv.org/pdf/1910.00694 


# load eye tracking model
if eye_model_type not in model_dict:
    print ("Model not found !!!")
    print ("valid models are:",list(model_dict.keys()))
    exit(1)
if useGPU:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
model = model_dict[eye_model_type]
model  = model.to(device)
filename = eye_model_path
if not os.path.exists(filename):
    print("eye model path not found")
    exit(1)
    
model.load_state_dict(torch.load(filename))
model = model.to(device)
model.eval()

# preprocess eye image
pilimg = Image.open(eye_image_path).convert("L")
H, W = pilimg.width , pilimg.height
#Fixed gamma value for      
table = 255.0*(np.linspace(0, 1, 256)**0.8)
pilimg = cv2.LUT(np.array(pilimg), table)
#local Contrast limited adaptive histogram equalization algorithm
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
img = clahe.apply(np.array(np.uint8(pilimg)))    
img = Image.fromarray(img)     
img = transform(img) 
# batchsize is 1
img = img.unsqueeze(0)  # Add batchsize dimension, resulting in shape [1, 1, 400, 640]
img = img.permute(0, 1, 3, 2)  # Permute dimensions to get shape [1, 1, 640, 400]
data = img.to(device)

# predict the segmentation
output = model(data)
predict = get_predictions(output) # integers labeling: pupil (label=3), iris (label=2), sclera (label=1) and background (label=0)


# for debugging, save the image with segmentation result
pred_img = predict[0].cpu().numpy()/3.0
inp = img[0].squeeze() * 0.5 + 0.5
img_orig = np.clip(inp,0,1)
img_orig = np.array(img_orig)
combine = np.hstack([img_orig,pred_img])
plt.imsave('eye_seg_pred.png',combine)

# 2. from the segmentation result, estimate gaze direction and camera matrix
#     - the gaze estimation used is hmd-eyes: https://github.com/pupil-labs/hmd-eyes https://arxiv.org/pdf/2403.19768v1



# 3. render the 3d gaussian splatting from the camera matrix

# TODO: set camera matrix from eye tracking
            
# Initialize system state (RNG)
safe_state(args.quiet)
# render
render_sets(splat_model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

