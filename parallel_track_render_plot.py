# File: home/ubuntu/gaussian_splatting_with_eye_tracking/parallel_track_render_plot.py
import json
import numpy as np
import matplotlib.pyplot as plt


# add an argument to deal with the scenaario where the timing of first image for foveal net is saved separately
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--first_image_separate', action='store_true', help='If the timing of the first image is saved separately')
args = parser.parse_args()

# Load timing data from JSON file
with open('timing_data.json', 'r') as f:
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
labels = ['Preprocess', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
colors = ['grey', 'red', 'green', 'blue', 'yellow']

# Print average times
print("Average time per step for the first eye image:")
for i in range(max_steps):
    print(f"Step {i} ({labels[i]}): {avg_step_times_first_image[i]:.2f} ms")

print("\nAverage time per step for the other 49 images:")
for i in range(max_steps):
    print(f"Step {i} ({labels[i]}): {avg_step_times_other_images[i]:.2f} ms")


plt.figure(figsize=(5, 6))
bar_width = 0.15
x_positions = np.arange(1)  # Only one bar per group

# First, plot for the first eye image
bottom = 0
for i in range(max_steps):
    plt.bar(
        x_positions[0], avg_step_times_first_image[i], bar_width,
        bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
    )
    bottom += avg_step_times_first_image[i]

# Then, plot for the other 49 images next to it
bottom = 0
for i in range(max_steps):
    plt.bar(
        x_positions[0] + bar_width + 0.05, avg_step_times_other_images[i], bar_width,
        bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
    )
    bottom += avg_step_times_other_images[i]

# Customize plot
# plt.xlabel('Image Group')
plt.ylabel('Latency (ms) averaged over 100 views $\\times$ 50 gazes')
# plt.title('Average Rendering Time per Step')

# Set x-ticks to label the bars
plt.xticks([x_positions[0] , x_positions[0] + 1 * bar_width +0.05], ['First rendering', 'Gaze updates'])

# Create a custom legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
plt.legend(handles, labels)

plt.tight_layout()
plt.savefig('rendering_time_per_step.png')
plt.show()



# next, plot fovealnet timing

# Load timing data from JSON file
with open('fovealnet_timing_data.json', 'r') as f:
    data = json.load(f)

# Assuming data structure:
# data = {
#     'inference_times': list of total inference times per image,
#     'layer_timings_per_image': list of [layer_times] per image
# }

if args.first_image_separate:
    with open('fovealnet_timing_data_first_image.json', 'r') as f:
        data_first_image = json.load(f)
    inference_times = np.array(data['inference_times'])  # Shape: (num_sequences * num_images-1,)
    layer_timings_per_image = np.array(data['layer_timings_per_image'])  # Shape: (num_sequences * num_images-1, num_layers)
    inference_times_first_image = np.array(data_first_image['inference_times'])  # Shape: (num_sequences,)
    layer_times_first_image = np.array(data_first_image['layer_timings_per_image'])  # Shape: (num_sequences, num_layers)

    num_images_per_sequence = 49  # Assuming each sequence has 50 images (1 first image + 49 updates)
    num_layers = layer_timings_per_image.shape[1]
    # Reshape arrays to (num_sequences, num_images_per_sequence, ...)
    num_sequences = inference_times.shape[0] // num_images_per_sequence
    inference_times_other_images = inference_times.reshape(num_sequences, num_images_per_sequence)
    layer_times_other_images = layer_timings_per_image.reshape(num_sequences, num_images_per_sequence, num_layers)
else:
    # Convert lists to numpy arrays for easier processing
    inference_times = np.array(data['inference_times'])  # Shape: (num_sequences * num_images,)
    layer_timings_per_image = np.array(data['layer_timings_per_image'])  # Shape: (num_sequences * num_images, num_layers)

    # Since images are processed sequentially, we can reshape the arrays based on the number of images per sequence
    num_images_per_sequence = 50  # Assuming each sequence has 50 images (1 first image + 49 updates)
    num_layers = layer_timings_per_image.shape[1]

    # Reshape arrays to (num_sequences, num_images_per_sequence, ...)
    num_sequences = inference_times.shape[0] // num_images_per_sequence
    inference_times = inference_times.reshape(num_sequences, num_images_per_sequence)
    layer_timings_per_image = layer_timings_per_image.reshape(num_sequences, num_images_per_sequence, num_layers)

    # Separate the first eye image and the other 49 images
    inference_times_first_image = inference_times[:, 0]  # Shape: (num_sequences,)
    inference_times_other_images = inference_times[:, 1:]  # Shape: (num_sequences, num_images_per_sequence - 1)

    layer_times_first_image = layer_timings_per_image[:, 0, :]  # Shape: (num_sequences, num_layers)
    layer_times_other_images = layer_timings_per_image[:, 1:, :]  # Shape: (num_sequences, num_images_per_sequence - 1, num_layers)

# Calculate average inference time for the first eye image over all sequences
avg_inference_time_first_image = np.mean(inference_times_first_image)
avg_layer_times_first_image = np.mean(layer_times_first_image, axis=0)  # Shape: (num_layers,)

# Calculate average inference time for the other 49 images over all sequences and images
avg_inference_time_other_images = np.mean(inference_times_other_images)
avg_layer_times_other_images = np.mean(layer_times_other_images, axis=(0, 1))  # Shape: (num_layers,)

# Print average times
print(f"Average inference time for the first eye image: {avg_inference_time_first_image:.2f} ms")
print("Average time per layer for the first eye image:")
for i in range(num_layers):
    if i == 0:
        print(f"Patch Embedding: {avg_layer_times_first_image[i]:.2f} ms")
    elif i == num_layers - 1:
        print(f"Final Layers: {avg_layer_times_first_image[i]:.2f} ms")
    else:
        print(f"Transformer Block {i}: {avg_layer_times_first_image[i]:.2f} ms")

print(f"\nAverage inference time for the other 49 images: {avg_inference_time_other_images:.2f} ms")
print("Average time per layer for the other 49 images:")
for i in range(num_layers):
    if i == 0:
        print(f"Patch Embedding: {avg_layer_times_other_images[i]:.2f} ms")
    elif i == num_layers - 1:
        print(f"Final Layers: {avg_layer_times_other_images[i]:.2f} ms")
    else:
        print(f"Transformer Block {i}: {avg_layer_times_other_images[i]:.2f} ms")

# Plotting the average layer times as stacked bar charts
labels = []
colors = []
for i in range(num_layers):
    if i == 0:
        labels.append('Patch Embedding')
        colors.append('grey')
    elif i == num_layers - 1:
        labels.append('Final Layers')
        colors.append('yellow')
    else:
        labels.append(f'Transformer Block {i}')
        colors.append(plt.get_cmap('tab20')(i))

plt.figure(figsize=(5, 6))
bar_width = 0.15
x_positions = np.arange(1)  # Only one bar per group

# First, plot for the first eye image
bottom = 0
for i in range(num_layers):
    plt.bar(
        x_positions[0], avg_layer_times_first_image[i], bar_width,
        bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
    )
    bottom += avg_layer_times_first_image[i]

# Then, plot for the other 49 images next to it
bottom = 0
for i in range(num_layers):
    plt.bar(
        x_positions[0] + bar_width + 0.05, avg_layer_times_other_images[i], bar_width,
        bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
    )
    bottom += avg_layer_times_other_images[i]

# Customize plot
plt.ylabel('Latency (ms) averaged over 100 sequences $\\times$ 50 eye images')

# Set x-ticks to label the bars
plt.xticks([x_positions[0], x_positions[0] + 1 * bar_width + 0.05], ['First inference', 'Gaze updates'])

# Create a custom legend
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(num_layers)]
plt.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.savefig('fovealnet_time_per_layer.png')
plt.show()