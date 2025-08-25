import matplotlib.pyplot as plt


def plot_fields(input, output, target, save_path, index=1):
    input_print = input[index].detach().cpu()
    output_print = output[index].detach().cpu()
    target_print = target[index].detach().cpu()

    vmin = target_print.min()
    vmax = target_print.max()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    c0 = ax0.pcolormesh(input_print)
    fig.colorbar(c0, ax=ax0)
    c1 = ax1.pcolormesh(output_print, vmin=vmin, vmax=vmax)
    fig.colorbar(c1, ax=ax1)
    c2 = ax2.pcolormesh(target_print, vmin=vmin, vmax=vmax)
    fig.colorbar(c2, ax=ax2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)