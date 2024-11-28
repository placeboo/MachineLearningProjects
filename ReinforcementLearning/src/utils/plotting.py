import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def set_plot_style(multiplier=1):
    """Set the default plot style"""
    plt.style.use('default')
    sns.set_style("whitegrid")

    plt.rcParams.update({
        'figure.figsize': (4 * multiplier, 3 * multiplier),  # Single column width for IEEE
        'font.size': 12 * multiplier,
        'axes.labelsize': 14 * multiplier,
        'axes.titlesize': 14 * multiplier,
        'xtick.labelsize': 13 * multiplier,
        'ytick.labelsize': 13 * multiplier,
        'legend.fontsize': 12 * multiplier,
        'figure.facecolor': 'white',
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{newtxmath}'
    })



def create_param_heatmap(data: pd.DataFrame,
                         x_param: str,
                         y_param: str,
                         metric: str,
                         title=None,
                         figsize=(10, 8),
                         cmap='YlOrRd',
                         annot=True,
                         fmt=".2f",
                         multiplier = 1):
    """Create a heatmap for two hyperparameters and performance metrics"""
    heatmap_df = data.pivot_table(
        index = y_param,
        columns = x_param,
        values = metric,
        aggfunc = 'mean'
    ).sort_index(ascending=False)
    set_plot_style(multiplier)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(heatmap_df,
                annot=annot,
                cbar_kws={'label': metric},
                fmt=fmt,
                cmap=cmap,
                annot_kws={'rotation': 0},
                ax=ax)

    if title:
        ax.set_title(title)
    # make integer ticks
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    fig.tight_layout()
    return fig, ax


def create_v_iters_plot(v_arr: np.ndarray,
                        title=None,
                        y_label=None,
                        figsize=(4,3),
                        linewidth=1):
    """Create a line plot for value iteration convergence"""
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, len(v_arr)+1)
    plt.plot(x, v_arr, linewidth=linewidth)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlabel("Iterations")
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    return fig, ax


def modified_plot_policy(val_max,
                         directions,
                         title=None,
                         figsize=(10, 8),
                         multiplier=1):
    """Plot the policy learned."""
    set_plot_style(multiplier)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        val_max,
        annot=directions,
        fmt="",
        cmap=sns.color_palette("magma_r", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        annot_kws={"size": 12*multiplier},
    ).set(title=title)
    return fig, ax


def save_plot(fig, dir, filename):
    os.makedirs(dir, exist_ok=True)
    filename = filename + '.png'
    file_path = os.path.join(dir, filename)
    fig.savefig(file_path, bbox_inches='tight', dpi=600)

