import matplotlib.pyplot as plt
import numpy as np


def vis_imfs(imfs, num_samples=500):
    """
        Visualizes the Intrinsic Mode Functions (IMFs) for each channel.

        Parameters:
          imfs (numpy.ndarray): A 3D array of shape (num_channels, num_imfs, num_data_points)
                                containing the IMFs for each channel.
          num_samples (int): The number of samples to display for each IMF.
                             Defaults to 500. If None or if greater than the available data points,
                             the full signal is plotted.

        Displays:
          A matplotlib figure with subplots arranged with channels as rows and IMFs as columns.
    """
    num_channels, num_imfs, num_data_points = imfs.shape

    if num_samples is None or num_samples > num_data_points:
        num_samples = num_data_points

    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    fig, axes = plt.subplots(num_channels, num_imfs, figsize=(5 * num_imfs, 3 * num_channels), constrained_layout=True)
    fig.suptitle("Channel IMFs", fontsize=20)

    for channel_index in range(num_channels):
        for i in range(num_imfs):
            ax = axes[channel_index, i] if num_imfs > 1 else axes[channel_index]
            ax.plot(imfs[channel_index, i, :num_samples])

            ax.set_ylabel(f'IMF {i+1} Amplitude', fontsize=10)

            if i == 0:
                ax.set_ylabel(f'Channel {channel_index + 1}\nIMF {i + 1} Amplitude', fontsize=10)

            if channel_index == 0:
                ax.set_title(f'IMF {i + 1}' if i < (num_imfs - 1) else 'Residual', fontsize=12)
            ax.set_xlim(0, num_samples)
            ax.set_xlabel('Samples', fontsize=10)  # Add x-axis label
            ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(pad=2.0)
    plt.show()


def vis_signal(signal):
    """
        Visualizes the original multi-channel signal.

        Parameters:
          signal (numpy.ndarray): A 2D array of shape (num_channels, num_data_points)
                                  representing the original signal for each channel.

        Displays:
          A single matplotlib plot overlaying each channel of the signal.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    num_channels, num_data_points = signal.shape
    t = np.arange(num_data_points)

    for channel_index in range(num_channels):
        ax.plot(t, signal[channel_index, :], label=f'Channel {channel_index + 1}')

    ax.set_title("Original Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    plt.show()
