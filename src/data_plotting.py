import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_formatting import startup_plotting

startup_plotting()

# --- Load data
df = pd.read_csv("AMME5710_lighting_data.csv")
print(df.head())

# --- Group by Lux and compute mean values
df_grouped = df.groupby("Lux", as_index=False).mean()

# --- Extract data
lux = df_grouped["Lux"]
chamfer = df_grouped["Best Chamfer Distance"]
hausdorff = df_grouped["Best Hausdorff"]

# --- Bar plot setup
x = np.arange(len(lux))  # positions
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, chamfer, width, label="Chamfer Distance", color="tab:blue")
bars2 = ax.bar(x + width/2, hausdorff, width, label="Hausdorff Distance", color="tab:orange")

# --- Axis labels and styling
ax.set_xlabel("Illuminance (Lux)")
ax.set_ylabel("Distance (m)")
ax.set_title("Chamfer and Hausdorff Distances vs. Lighting Conditions")
ax.set_xticks(x)
ax.set_xticklabels(lux)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)

# --- Optionally annotate bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.001, f"{height:.3f}",
            ha="center", va="bottom", fontsize=11)

plt.tight_layout()

plt.savefig("Plot of Chamfer and Hausdorff Distance")
plt.show()








# --- Load data
df = pd.read_csv("AMME5710_foil_data.csv")
print(df.head())

# --- Group by Lux and compute mean values
df_grouped = df.groupby("Trial", as_index=False).mean()

# --- Extract data
lux = df_grouped["Trial"]
chamfer = df_grouped["Best Chamfer Distance"]
hausdorff = df_grouped["Best Hausdorff"]

# --- Bar plot setup
x = np.arange(len(lux))  # positions
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, chamfer, width, label="Chamfer Distance", color="tab:blue")
bars2 = ax.bar(x + width/2, hausdorff, width, label="Hausdorff Distance", color="tab:orange")

# --- Axis labels and styling
ax.set_xlabel("Trial")
ax.set_ylabel("Distance (m)")
ax.set_title("Chamfer and Hausdorff Distances with Foil Encasing")
ax.set_xticks(x)
ax.set_xticklabels(lux)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)

# --- Optionally annotate bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.001, f"{height:.3f}",
            ha="center", va="bottom", fontsize=11)

plt.tight_layout()

plt.savefig("Plot of Chamfer and Hausdorff Distance with Foil")
plt.show()
