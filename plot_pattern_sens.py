import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pdb
from tabulate import tabulate
import os
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib import rc

error_pats = ["00", "01", "10", "11"]
des_pats = ["00", "01", "10", "11"]

model = "resnet18"
base_acc = 90.689
heat_map = np.empty((4, 4))

for pat_index, error_pat in enumerate(error_pats):
    for des_index, error_des in enumerate(des_pats):
        if error_des == error_pat:
            heat_map[pat_index, des_index] = 0
            continue
        else:
            acc = pd.read_csv(f"./result/Pattern-{error_pat}-to-{error_des}.csv")
            for index, i in enumerate(acc["Acc."]):
                if i < (base_acc - 20.0):
                    heat_map[pat_index, des_index] = index
                    break
                else:
                    continue

with plt.style.context(["ieee", "no-latex"]):
    mpl.rcParams['font.family'] = 'NimbusRomNo9L'
    fig, ax = plt.subplots(figsize=(2, 2))
    heat_map = 24 - heat_map
    im = ax.imshow(heat_map, cmap="OrRd")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Time (s)", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(error_pats)))
    ax.set_yticks(np.arange(len(des_pats)))

    ax.set_xticklabels(error_pats, fontsize=8)
    ax.set_yticklabels(des_pats, fontsize=8)

    ax.set_xlabel("Drifted pattern", fontsize=8)
    ax.set_ylabel("Original pattern", fontsize=8)

plt.tight_layout()
fig.savefig("./Figures/pattern_heatmap.svg", dpi=300)
# os.system("zathura pattern_heatmap.pdf")
# plt.show()


