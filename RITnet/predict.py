# test: given an input image, output a predicted feature

import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions


# if __name__ == '__main__':
    
args = parse_args()

if args.model not in model_dict:
    print ("Model not found !!!")
    print ("valid models are:",list(model_dict.keys()))
    exit(1)

if args.useGPU:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
    
model = model_dict[args.model]
model  = model.to(device)
filename = args.load
if not os.path.exists(filename):
    print("model path not found !!!")
    exit(1)
    
model.load_state_dict(torch.load(filename))
model = model.to(device)
model.eval()


