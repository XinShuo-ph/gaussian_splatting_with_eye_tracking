# collect the prediction errors for all layers
# the test data are not well labeled, use validation data instead
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from timm_vit import VisionTransformerFoveated
from utils import EdsDataset
import pandas as pd
import os
from tqdm import tqdm
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to test data and model
epoch_idx = 26
# test_folder = '/home/ubuntu/openeds/train'
# test_info = '/home/ubuntu/openeds/train.csv'
test_folder = '/home/ubuntu/openeds/test'
test_info = '/home/ubuntu/openeds/test.csv'
model_path = '/home/ubuntu/gaussian_splatting_with_eye_tracking/fovealnet/results_epoch/epoch_%d/model_epoch_%d.pt'%(epoch_idx, epoch_idx)

import pandas as pd

# Read the CSV file directly
test_info_df = pd.read_csv(test_info)

# Filter the DataFrame for seq_name < 100 and image_name < 50
# filtered_df = test_info_df[test_info_df['image'].apply(lambda x: int(x.split('\\')[0]) < 7000 and int(x.split('\\')[0]) >= 6400 and int(x.split('\\')[1]) < 50)]
filtered_df = test_info_df[test_info_df['image'].apply(lambda x: int(x.split('\\')[0]) < 4700 and int(x.split('\\')[0]) >= 4500 and int(x.split('\\')[1]) < 50)]

# Write the filtered DataFrame to a new CSV file
filtered_df.to_csv('train_subset.csv', index=False)

# Load the test dataset with the new info file
test_dataset = EdsDataset(image_folder=test_folder, info_file='train_subset.csv')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
model = VisionTransformerFoveated(num_layers=6, top_k=1.0)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

errors = []

with torch.no_grad():
    for images, _, _, gaze_gt_vecs in tqdm(test_loader, desc="Processing test data"):
        images = images.to(device)
        gaze_gt_vecs = gaze_gt_vecs.to(device)
        outputs = model(images)  # outputs shape: (6, batch_size, 2)
        outputs = outputs.squeeze(1)  # Shape: (6, 2)
        differences = outputs - gaze_gt_vecs.squeeze(0)  # Shape: (6, 2)
        errors.append(differences.cpu().numpy())

errors = np.array(errors)  # Shape: (N, 6, 2)
print("Errors shape:", errors.shape)
# Save the errors array to a file
np.save('error_stat/prediction_errors.npy', errors)

# plot all the 2*6 12 distributions in one plot, label layer idx and gaze x/y
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.figure(figsize=(12, 8))
ax = plt.gca()

for i in range(6):
    # Calculate the density for x
    density_x = gaussian_kde(errors[:, i, 0])
    xs = np.linspace(min(errors[:, i, 0]), max(errors[:, i, 0]), 1000)
    ax.plot(xs, density_x(xs), label=f"Layer {i} x")

    # Calculate the density for y
    density_y = gaussian_kde(errors[:, i, 1])
    ys = np.linspace(min(errors[:, i, 1]), max(errors[:, i, 1]), 1000)
    ax.plot(ys, density_y(ys), label=f"Layer {i} y")

ax.set_title("Error Distributions for Layers")
ax.set_xlabel("Error Value")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig('error_stat/prediction_errors.png')
plt.show()
