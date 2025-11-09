# Save this as plot_formatting.py
from cycler import cycler
import matplotlib.pyplot as plt

# Provided by Jack Naylor (tutor from previous engineering subjects) and edited as needed

import matplotlib as mpl

mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.25

def startup_plotting(font_size=22, line_width=1.5, output_dpi=600, tex_backend=False):
    if tex_backend:
        try:
            plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })
        except:
            print("WARNING: LaTeX backend not configured properly. Not using.")
            plt.rcParams.update({"font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })

    # Format lines, axes, grids, etc.
    plt.rcParams.update({
        "lines.linewidth": line_width,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.linewidth": 0.5,
        "axes.prop_cycle": cycler("color", [
            "#0072B2", "#E69F00", "#009E73", "#CC79A7", 
            "#56B4E9", "#D55E00", "#F0E442", "#000000"]),

        "grid.linewidth": 0.25,
        "grid.alpha": 0.5,

        "legend.framealpha": 0.7,
        "legend.edgecolor": [1, 1, 1],

        "savefig.dpi": output_dpi,
        "savefig.format": 'pdf'
    })

    # Change default font sizes.
    plt.rc('font', size=font_size)  # Controls default text size
    plt.rc('axes', titlesize=font_size)  # Font size of the title
    plt.rc('axes', labelsize=font_size)  # Font size of the x and y labels
    plt.rc('xtick', labelsize=0.8*font_size)  # Font size of the x tick labels
    plt.rc('ytick', labelsize=0.8*font_size)  # Font size of the y tick labels
    plt.rc('legend', fontsize=0.8*font_size)  # Font size of the legend
