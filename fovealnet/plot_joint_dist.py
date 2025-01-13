import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Load the errors array from the .npy file
errors = np.load('error_stat/prediction_errors.npy')  # Shape: (N, 6, 2)

# Use the 6th layer prediction errors (index 5)
# layer_index = 5

for layer_index in range(6):

    layer_errors = errors[:, layer_index, :]  # Shape: (N, 2)

    # change to degree
    layer_errors = np.rad2deg(layer_errors)

    # Extract x and y errors
    x_errors = layer_errors[:, 0]
    y_errors = layer_errors[:, 1]


    x_mean, x_std = norm.fit(x_errors)
    y_mean, y_std = norm.fit(y_errors)

    avgstd = (x_std + y_std) / 2

    # Define the axis limits
    xmin, xmax = -2.5*avgstd, 2.5*avgstd
    ymin, ymax = -2.5*avgstd, 2.5*avgstd

    # Define the number of bins
    bins = 20  # Adjust as needed for resolution

    # Create the figure and gridspec layout
    fig = plt.figure(figsize=(5, 5))
    grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)

    # Main plot: 2D histogram
    main_ax = fig.add_subplot(grid[1:4, 0:3])

    # Top histogram: x marginal
    x_hist_ax = fig.add_subplot(grid[0, 0:3], sharex=main_ax)

    # Right histogram: y marginal
    y_hist_ax = fig.add_subplot(grid[1:4, 3], sharey=main_ax)

    # Plot the 2D histogram on the main axes
    h = main_ax.hist2d(
        x_errors, y_errors,
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
        cmap='viridis',
        density=True
    )
    main_ax.set_xlabel('error of gaze prediction x ($^\\circ$)', fontsize=14)
    main_ax.set_ylabel('error of gaze prediction y ($^\\circ$)', fontsize=14)

    # Set axis limits for main axes
    main_ax.set_xlim(xmin, xmax)
    main_ax.set_ylim(ymin, ymax)

    # Plot the marginal distributions
    x_hist = x_hist_ax.hist(
        x_errors,
        bins=bins,
        range=(xmin, xmax),
        color='gray',
        density=True
    )
    y_hist = y_hist_ax.hist(
        y_errors,
        bins=bins,
        range=(ymin, ymax),
        orientation='horizontal',
        color='gray',
        density=True
    )

    # Gaussian fit for x errors
    x_fit = np.linspace(xmin, xmax, 100)
    x_pdf = norm.pdf(x_fit, x_mean, x_std)
    x_hist_ax.plot(x_fit, x_pdf, 'r--', label='Gaussian fit')
    x_hist_ax.legend( fontsize=12, loc = 'upper right')
    print('x_mean:', x_mean)
    print('x_std:', x_std)

    # Gaussian fit for y errors
    y_fit = np.linspace(ymin, ymax, 100)
    y_pdf = norm.pdf(y_fit, y_mean, y_std)
    y_hist_ax.plot(y_pdf, y_fit, 'r--', label='Gaussian fit')
    # y_hist_ax.legend()
    print('y_mean:', y_mean)
    print('y_std:', y_std)

    # Hide the spines and ticks for marginal histograms
    x_hist_ax.spines['right'].set_visible(False)
    x_hist_ax.spines['top'].set_visible(False)
    x_hist_ax.spines['left'].set_visible(False)
    x_hist_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    x_hist_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    y_hist_ax.spines['right'].set_visible(False)
    y_hist_ax.spines['top'].set_visible(False)
    y_hist_ax.spines['bottom'].set_visible(False)
    y_hist_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    y_hist_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Adjust limits of the histograms to match the main axes
    x_hist_ax.set_xlim(main_ax.get_xlim())
    y_hist_ax.set_ylim(main_ax.get_ylim())

    # Add colorbar to the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(h[3], cax=cbar_ax)
    cbar.set_label('Probability Density', fontsize=14)

    # Ensure the directory exists
    if not os.path.exists('error_stat'):
        os.makedirs('error_stat')

    # Save and show the plot
    # plt.savefig('error_stat/joint_distribution_layer5.png', dpi=300, bbox_inches='tight')
    plt.savefig('error_stat/joint_distribution_layer%d.pdf'%(layer_index+1), bbox_inches='tight')
    plt.savefig('error_stat/joint_distribution_layer%d.png'%(layer_index+1), bbox_inches='tight')
    plt.show()

# next, plot the avg_std of all layers
avgstds = []
for layer_index in range(6):
    layer_errors = errors[:, layer_index, :]  # Shape: (N, 2)
    layer_errors = np.rad2deg(layer_errors)
    x_errors = layer_errors[:, 0]
    y_errors = layer_errors[:, 1]
    x_mean, x_std = norm.fit(x_errors)
    y_mean, y_std = norm.fit(y_errors)
    print(layer_index+1, '&',x_std, '&',y_std,'\\\\')   
    avgstd = (x_std + y_std) / 2
    avgstds.append(avgstd)

plt.figure().set_size_inches(10, 2)
plt.semilogy(range(1,7),avgstds, marker='o')
plt.ylim([0.1, 10])
print(avgstds)
plt.xlabel('FovealNet layers', fontsize=10)
plt.ylabel('$\\sigma$ ($^\\circ$)', fontsize=10)
# plt.grid()
plt.savefig('error_stat/avg_std_layers.pdf', bbox_inches='tight')
plt.savefig('error_stat/avg_std_layers.png', bbox_inches='tight')
plt.show()