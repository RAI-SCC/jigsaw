import matplotlib.pyplot as plt
import numpy as np


def plot_fields(input, output, target, save_path, index):
    """Overloaded plotting function."""
    input_print = np.flip(input.detach().cpu().numpy(), axis=1)
    output_print = np.flip(output.detach().cpu().numpy(), axis=1)  # - input_print
    target_print = np.flip(target.detach().cpu().numpy(), axis=1)  # - input_print
    vmin = -2
    vmax = 2

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    c0 = ax0.pcolormesh(input_print[index], vmin=vmin, vmax=vmax)
    fig.colorbar(c0, ax=ax0)
    c1 = ax1.pcolormesh(output_print[index], vmin=vmin, vmax=vmax)
    fig.colorbar(c1, ax=ax1)
    c2 = ax2.pcolormesh(target_print[index], vmin=vmin, vmax=vmax)
    fig.colorbar(c2, ax=ax2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
