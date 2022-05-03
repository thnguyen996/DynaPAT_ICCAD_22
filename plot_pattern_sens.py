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
base_acc = 82.01
heat_map = np.empty((4, 4))

for pat_index, error_pat in enumerate(error_pats):
    for des_index, error_des in enumerate(des_pats):
        if error_des == error_pat:
            heat_map[pat_index, des_index] = 0.
            continue
        else:
            acc = pd.read_csv(f"./results-2022/{model}-test-cases-fixed-point-{error_pat}-{error_des}.csv")
            acc_print = acc["Acc."][13]
            print(f"{error_pat} --> {error_des}: {acc_print}")
            heat_map[pat_index, des_index] = base_acc - acc["Acc."][13]
            # for index, i in enumerate(acc["Acc."]):
            #     if i < (base_acc - 10.0):
            #         heat_map[pat_index, des_index] = index
            #         break
            #     else:
            #         continue

with plt.style.context(["ieee", "no-latex"]):
    mpl.rcParams['font.family'] = 'NimbusRomNo9L'
    fig, ax = plt.subplots(figsize=(2, 2))
    # heat_map = 24 - heat_map
    im = ax.imshow(heat_map, cmap="OrRd")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(" Accuracy Loss (%) ", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(error_pats)))
    ax.set_yticks(np.arange(len(des_pats)))
    ax.set_xticklabels(error_pats, fontsize=8)
    ax.set_yticklabels(des_pats, fontsize=8)
    ax.set_xlabel("Drifted Pattern", fontsize=8)
    ax.set_ylabel("Original Pattern", fontsize=8)

plt.tight_layout()
fig.savefig(f"./Figures/pattern_heatmap_{model}_fixed_point.pdf", dpi=300)
os.system(f"zathura ./Figures/pattern_heatmap_{model}_fixed_point.pdf")
# plt.show()

