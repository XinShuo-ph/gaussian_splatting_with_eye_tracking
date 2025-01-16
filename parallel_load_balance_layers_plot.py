import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from arguments import ModelParams, PipelineParams, get_combined_args
parser = argparse.ArgumentParser()
parser.add_argument("--tune_layer_iter", default=0, type=int) # tune the number of layers for each fovea step
parser.add_argument("--cpucount", default=3, type=int) # number of cpus
parser.add_argument("--scene", default='', type=str) # scene name, default is empty string
parser.add_argument("--resnet", action='store_true') # use resnet instead of vit
parser.add_argument("--pruned", action='store_true') # use pruned model
args = parser.parse_args()

if args.resnet:
    with open('tune_layers_it%d_cpu%d_%s_3DGS_timing_resnet34.json'%(args.tune_layer_iter, args.cpucount, args.scene), 'r') as f:
        data = json.load(f)
elif args.pruned:
    with open('tune_layers_it%d_cpu%d_%s_3DGS_timing_pruned.json'%(args.tune_layer_iter, args.cpucount, args.scene), 'r') as f:
        data = json.load(f)
elif args.scene == '': # in this case, read the old data (which is for truck)
    with open('tune_layers_it%d_cpu%d_3DGS_timing.json'%(args.tune_layer_iter, args.cpucount), 'r') as f:
        data = json.load(f)
else:
    with open('tune_layers_it%d_cpu%d_%s_3DGS_timing.json'%(args.tune_layer_iter, args.cpucount, args.scene), 'r') as f:
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



# this var is to be plotted
avg_step_times_first_image = np.mean(first_image_times, axis=0)  # Shape: (max_steps,)
print('avg_step_times_first_image:\n', avg_step_times_first_image)


# next, process the fovealnet timing data
if args.resnet:
    with open('tune_layers_it%d_cpu%d_%s_fovealnet_timing_resnet34.json'%(args.tune_layer_iter, args.cpucount, args.scene), 'r') as f:
        data = json.load(f)
elif args.pruned:
    with open('tune_layers_it%d_cpu%d_%s_fovealnet_timing_pruned.json'%(args.tune_layer_iter, args.cpucount, args.scene), 'r') as f:
        data = json.load(f)
elif args.scene == '': # in this case, read the old data (which is for truck)
    with open('tune_layers_it%d_cpu%d_fovealnet_timing.json'%(args.tune_layer_iter, args.cpucount), 'r') as f:
        data = json.load(f)
else:
    with open('tune_layers_it%d_cpu%d_%s_fovealnet_timing.json'%(args.tune_layer_iter, args.cpucount, args.scene), 'r') as f:
        data = json.load(f)

inference_times = np.array(data['inference_times'])  # Shape: (num_sequences * num_images,)
layer_timings_per_image = np.array(data['layer_timings_per_image'])  # Shape: (num_sequences * num_images, num_layers)

# Since images are processed sequentially, we can reshape the arrays based on the number of images per sequence
num_images_per_sequence = 1  # Assuming each sequence has 50 images (1 first image + 49 updates)
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

# this var is to be plotted
avg_layer_times_first_image = np.mean(layer_times_first_image, axis=0)  # Shape: (num_layers,)




###### next, using avg_step_times_first_image and avg_layer_times_first_image to compute the idle time

if args.tune_layer_iter == 1:
    num_layers -=1

idle_times = np.zeros_like(avg_step_times_first_image) 

# now using the timing data, run as if the two models are running in parallel and record the idle time
curstep = 2 # could be idle only after render step 2 (step 0: preprocess, step 1: lowest res)
render_time = avg_step_times_first_image[0] + avg_step_times_first_image[1] + avg_step_times_first_image[2]
curfovealstep = 0
fovealtime = avg_layer_times_first_image[0] + 0.0
print('simulating the timing data...')
while curstep < len(avg_step_times_first_image)-1:
    while fovealtime <= render_time and curfovealstep < num_layers-1:
        curfovealstep += 1
        fovealtime += avg_layer_times_first_image[curfovealstep]
    if curfovealstep == num_layers-1:
        max_rounds = curstep+1
        break
    if fovealtime < render_time:
        max_rounds = curstep+1
        break
    if fovealtime > render_time and fovealtime - avg_layer_times_first_image[curfovealstep] <= render_time - avg_step_times_first_image[curstep]:
        # idle only if current foveal latency > current render latency, and previous foveal latency <= previous render latency
        # i.e. last rendering receives last gaze prediction ( previous foveal latency <= previous render latency), now the rendering finishes but new gaze prediction is not ready yet (current foveal latency > current render latency)
        idle_times[curstep] = fovealtime - render_time
        curstep += 1
        render_time = fovealtime + avg_step_times_first_image[curstep]
    else:
        curstep += 1
        render_time += avg_step_times_first_image[curstep]
    print('curstep:', curstep, 'curfovealstep:', curfovealstep, 'render_time:', render_time, 'fovealtime:', fovealtime, 'idle:', idle_times[curstep-1])
    # if max_rounds:
    # if max_rounds is defined, print it
    if 'max_rounds' in locals():
        print('max_rounds:', max_rounds)
    

    if curstep == len(avg_step_times_first_image)-1:
        max_rounds = curstep+1

        

print('idle_times:\n', idle_times)
print('max_rounds:\n', max_rounds)







# Plotting the average step times as stacked bar charts

labels = ['Preprocess'] + ['round %d'%i for i in range(1, max_rounds)] + ['idle']
# instead of manually picking color, use colormap 'jet'

colors = plt.cm.jet(np.linspace(0, 0.9, max_rounds-1))
colors = ['grey'] + list(colors) + ['black']

# colors = ['grey', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'black']


plt.figure(figsize=(5, 6))
bar_width = 0.15
x_positions = np.arange(1)  # Only one bar per group

# also write the step-wise lapse time to file, so that I can copy to an excel
if args.resnet:
    ftxt = open('tune_layers_it%d_cpu%d_%s_resnet34.txt'%(args.tune_layer_iter, args.cpucount, args.scene), 'w')
elif args.pruned:
    ftxt = open('tune_layers_it%d_cpu%d_%s_pruned.txt'%(args.tune_layer_iter, args.cpucount, args.scene), 'w')
else:
    ftxt = open('tune_layers_it%d_cpu%d_%s.txt'%(args.tune_layer_iter, args.cpucount, args.scene), 'w')

ftxt.write('Steps: ')
for labelname in ['Preprocess'] + ['round_%d'%i for i in range(1, max_rounds)] + ['idle']:
    if labelname != 'idle':
        ftxt.write(labelname + ' ')
    else:
        ftxt.write('\n')

ftxt.write('3DGS: ')
# First, plot for the first eye image
bottom = 0
for i in range(max_rounds):
    # plot render time
    if i == max_rounds-1:
        plt.bar(
            x_positions[0], avg_step_times_first_image[-1], bar_width,
            bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
        )
        bottom += avg_step_times_first_image[-1]
        ftxt.write(str(avg_step_times_first_image[-1]) + '\n')
    else:
        plt.bar(
            x_positions[0], avg_step_times_first_image[i], bar_width,
            bottom=bottom, color=colors[i], label=labels[i] if i == 0 else ""
        )
        bottom += avg_step_times_first_image[i]
        ftxt.write(str(avg_step_times_first_image[i]) + ' ')
    # plot idle time
    if idle_times[i] > 0:
        plt.bar(
            x_positions[0], idle_times[i], bar_width,
            bottom=bottom, color=colors[-1], label=labels[i] if i == 0 else ""
        )
    bottom += idle_times[i]

ftxt.write('idle: ')
for i in range(max_rounds):
    ftxt.write(' ' + str(idle_times[i]) )
ftxt.write('\n')

# Create a custom legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
# plt.legend(handles, labels, loc = 'lower left')
first_legend = plt.legend(handles, labels, loc='lower left', fontsize=12)
plt.gca().add_artist(first_legend)  # Add first legend back to plot



# then plot the fovealnet layer times as a second bar with its own colors

# instead of manually picking color, use colormap 'cool'

colors_fovealnet = plt.cm.cool(np.linspace(0, 1, num_layers-2))
colors_fovealnet = ['grey'] + list(colors_fovealnet) 
if args.resnet:
    labels_fovealnet = ['Embedding'] + ['ResNet Layer %d'%i for i in range(1, num_layers-1)]
else:
    labels_fovealnet = ['Embedding'] + ['ViT Layer %d'%i for i in range(1, num_layers-1)]

ftxt.write('Steps: ')
if args.resnet:
    for labelname in ['Embedding'] + ['ResNet_Layer_%d'%i for i in range(1, num_layers-1)]:
        ftxt.write(labelname + ' ')
else:
    for labelname in ['Embedding'] + ['ViT_Layer_%d'%i for i in range(1, num_layers-1)]:
        ftxt.write(labelname + ' ')
ftxt.write('\n')

bottom = 0
if args.resnet:
    ftxt.write('ResNet: ')
else:
    ftxt.write('FovealNet: ')
for i in range(num_layers-1): # the last layer is not useful because each layer already does the FC layers to output gaze prediction
    plt.bar(
        x_positions[0] + bar_width + 0.05, avg_layer_times_first_image[i], bar_width,
        bottom=bottom, color=colors_fovealnet[i], label='Layer %d'%i if i == 0 else ""
    )
    bottom += avg_layer_times_first_image[i]
    ftxt.write(str(avg_layer_times_first_image[i]) + ' ')
ftxt.write('\n')

# add a separate legend for the fovealnet layers
handles_fovealnet = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors_fovealnet]
plt.legend(handles_fovealnet, labels_fovealnet, loc='lower right', fontsize=12)  # Add second legend

# Set x-ticks to label the bars
if args.resnet:
    plt.xticks([x_positions[0], x_positions[0] + 1 * bar_width + 0.05], ['3DGS', 'ResNet'], fontsize=16)
else:
    plt.xticks([x_positions[0], x_positions[0] + 1 * bar_width + 0.05], ['3DGS', 'FovealNet'], fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('Latency (ms)', fontsize=16)
if args.resnet:
    plt.ylim(0,40)
else:
    plt.ylim(0,30)
plt.tight_layout()
if args.resnet:
    plt.savefig('tune_layers_it%d_cpu%d_%s_resnet34.png'%(args.tune_layer_iter, args.cpucount, args.scene))
    plt.savefig('tune_layers_it%d_cpu%d_%s_resnet34.pdf'%(args.tune_layer_iter, args.cpucount, args.scene), bbox_inches='tight')
elif args.pruned:
    plt.savefig('tune_layers_it%d_cpu%d_%s_pruned.png'%(args.tune_layer_iter, args.cpucount, args.scene))
    plt.savefig('tune_layers_it%d_cpu%d_%s_pruned.pdf'%(args.tune_layer_iter, args.cpucount, args.scene), bbox_inches='tight')
else:
    plt.savefig('tune_layers_it%d_cpu%d_%s.png'%(args.tune_layer_iter, args.cpucount, args.scene))
    plt.savefig('tune_layers_it%d_cpu%d_%s.pdf'%(args.tune_layer_iter, args.cpucount, args.scene), bbox_inches='tight')
plt.show()

