import numpy as np
import matplotlib.pyplot as plt

# read topk_tune_log_cpu2.txt
timing_3DGS, timing_fovealnet, topk = np.loadtxt('topk_tune_log_cpu2.txt', unpack=True)

fig, ax1 = plt.subplots(figsize=(6, 6))

# left ticks show timing, range from 20ms to 30ms
ax1.set_ylim(0, 40)
ax1.set_ylabel('Latency (ms)', fontsize=16)
ax1.plot(timing_3DGS, 'b', marker = "^", label='3DGS latency')
ax1.plot(timing_fovealnet, 'r', marker = "o", label='FovealNet latency')

# Create a second y-axis for topk
ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
ax2.set_ylabel('pruning ratio', fontsize=16)
ax2.plot(topk, 'g',markersize=5, marker = 's', label='pruning ratio')

# add legend for solid and dashed lines
ax2.plot([1e9,1e9], color='grey',label = '2 CPUs')
ax2.plot([1e9,1e9],':', color='grey',label = '3 CPUs')


# plot the same thing for 3 CPUs

timing_3DGS, timing_fovealnet, topk = np.loadtxt('topk_tune_log_cpu3.txt', unpack=True)
ax1.plot(timing_3DGS, 'b:', marker = "^")
ax1.plot(timing_fovealnet, 'r:', marker = "o")
ax2.plot(topk, 'g:',markersize=5, marker = 's')

# horizontal axis is iterations, from 0 to 20
ax1.set_xlim(0, len(timing_3DGS))
ax1.set_xlabel('Iterations', fontsize=16)

# add legend
# ax1.legend(loc='upper left', fontsize=16)
# ax2.legend(loc='upper right', fontsize=16)

# Instead of separate legends, get lines and labels from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Combine lines and labels from both axes
lines = lines1 + lines2
labels = labels1 + labels2

# Create a single legend
ax1.legend(lines, labels, loc='lower left', fontsize=16)

# set ticks font size
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)

# save the plot
plt.savefig('topk_tune.pdf', bbox_inches='tight')

