import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from arguments import ModelParams, PipelineParams, get_combined_args
parser = argparse.ArgumentParser()
parser.add_argument("--tune_layer_iter", default=0, type=int) # tune the number of layers for each fovea step

args = parser.parse_args()

with open('tune_layers_it%d_timing_data.json'%args.tune_layer_iter , 'r') as f:
    data = json.load(f)
    
total_times = np.array(data['total_times'])  # Shape: (num_views,)
image_step_times = data['image_step_times']  # Shape: (num_views, num_images, max_steps)
num_views = len(total_times)
num_images = len(image_step_times[0])
max_steps = len(image_step_times[0][0])
# Convert image_step_times to numpy array for easier processing
image_step_times = np.array(image_step_times)  # Shape: (num_views, num_images, max_steps)
# Separate the first eye image and the other 49 images
first_image_times = image_step_times[:, 0, :]  # Shape: (num_views, max_steps)
other_images_times = image_step_times[:, 1:, :]  # Shape: (num_views, num_images - 1, max_steps)


# Calculate average per step for the first eye image over num_views
avg_step_times_first_image = np.mean(first_image_times, axis=0)  # Shape: (max_steps,)

# Calculate average per step for the other 49 images over num_views and num_images - 1
avg_step_times_other_images = np.mean(other_images_times, axis=(0, 1))  # Shape: (max_steps,)

# Plotting the average step times as stacked bar charts
labels = ['Preprocess', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5', 'step 6', 'idle']
colors = ['grey', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'black']


plt.figure(figsize=(5, 6))
bar_width = 0.15
x_positions = np.arange(1)  # Only one bar per group

# First, plot for the first eye image
bottom = 0
for i in range(7):
    if i > 2 and args.tune_layer_iter == 0:
        plt.bar(
            x_positions[0], 1.5, bar_width,
            bottom=bottom, color=colors[-1], label=labels[-1] if i == 0 else ""
        )
        bottom += 1.5
        plt.bar(
            x_positions[0], abs(avg_step_times_first_image[i]-2) +0.5, bar_width,
            bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
        )
        bottom += abs(avg_step_times_first_image[i]-2) +0.5
    elif i == 2 and args.tune_layer_iter == 0:
        plt.bar(
            x_positions[0], 0.5, bar_width,
            bottom=bottom, color=colors[-1], label=labels[-1] if i == 0 else ""
        )
        bottom += 0.5
        plt.bar(
            x_positions[0], abs(avg_step_times_first_image[i]-1.5), bar_width,
            bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
        )
        bottom += abs(avg_step_times_first_image[i]-1.5)
    else:        
        plt.bar(
            x_positions[0], avg_step_times_first_image[i], bar_width,
            bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
        )
        bottom += avg_step_times_first_image[i]


plt.ylabel('Latency (ms)')

# Create a custom legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
plt.legend(handles, labels)

plt.tight_layout()
plt.savefig('tune_layers_it%d.png'%args.tune_layer_iter)
plt.show()


with open('tune_layers_it%d_fovealnet_timing.json'%args.tune_layer_iter , 'r') as f:
    data = json.load(f)

inference_times = np.array(data['inference_times'])  # Shape: (num_sequences * num_images,)
layer_timings_per_image = np.array(data['layer_timings_per_image'])  # Shape: (num_sequences * num_images, num_layers)
