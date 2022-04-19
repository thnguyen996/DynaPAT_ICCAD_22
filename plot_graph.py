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

def plot_graph(
    time, data_list: list, save_fig=False, file_name=None, save_dir=None
):
    with plt.style.context(["ieee", "no-latex"]):
        mpl.rcParams['font.family'] = 'NimbusRomNo9L'

        fig, ax = plt.subplots(figsize=(2.2, 2.2))

        data = data_list
        method_name = data["Method name"]
        data_value = data["data"]
        color = data["color"]
        if "marker" in data:
            marker = data["marker"]
            ax.plot(time, data_value, linewidth=1.7, markevery=3, markerfacecoloralt='none', markersize=4, color=color, label=method_name)
        else:
            ax.plot(time, data_value, linewidth=1.7, color=color, label=method_name)

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Classification Accuracy (%)", fontsize=8)
        ax.set_xticks(time[0:25:4])
        ax.set_xticklabels(
            (
                "$\mathregular{2^{1}}$",
                "$\mathregular{2^{5}}$",
                "$\mathregular{2^{9}}$",
                "$\mathregular{2^{13}}$",
                "$\mathregular{2^{17}}$",
                "$\mathregular{2^{21}}$",
                "$\mathregular{2^{25}}$",
                )
        )
        # ax.legend(loc=0, prop={"size": 8})
        # plt.tight_layout()
        if save_fig and save_dir is not None:
            fig.savefig(save_dir + file_name + ".svg", dpi=300)
        # os.system(f"junest mupdf {save_dir}{file_name}.pdf")
        plt.show()

def plot_graph_1year(
    time, data_list: list, save_fig=False, file_name=None, save_dir=None
):
    with plt.style.context(["ieee", "no-latex"]):
        mpl.rcParams['font.family'] = 'NimbusRomNo9L'

        fig, ax = plt.subplots(figsize=(2, 2))

        for data in data_list:
            method_name = data["Method name"]
            data_value = data["data"]
            linestyle = data["style"]
            color = data["color"]
            if "marker" in data:
                marker = data["marker"]
                ax.plot(time, data_value, linestyle, linewidth=1.7, marker=marker, markevery=3, markerfacecoloralt='none', markersize=4, color=color, label=method_name)
            else:
                ax.plot(time, data_value, linestyle, linewidth=1.4, color=color, label=method_name)

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Classification Accuracy (%)", fontsize=8)
        ax.set_xticks(time[0:25:4])
        ax.set_xticklabels(
            (
                "$\mathregular{2^{1}}$",
                "$\mathregular{2^{5}}$",
                "$\mathregular{2^{9}}$",
                "$\mathregular{2^{13}}$",
                "$\mathregular{2^{17}}$",
                "$\mathregular{2^{21}}$",
                "$\mathregular{2^{25}}$",
                )
        )
        ax.legend(loc=0, prop={"size": 8})
        plt.tight_layout()
        if save_fig and save_dir is not None:
            fig.savefig(save_dir + file_name + ".pdf", dpi=300)
        os.system(f"zathura {save_dir}{file_name}.pdf")

def plot_graph_1year_imagenet(
    time, data_list: list, save_fig=False, file_name=None, save_dir=None
):
    with plt.style.context(["ieee", "no-latex"]):
        mpl.rcParams['font.family'] = 'NimbusRomNo9L'

        fig, ax = plt.subplots(figsize=(1.8, 1.8))

        for data in data_list:
            method_name = data["Method name"]
            data_value = data["data"]
            linestyle = data["style"]
            color = data["color"]
            if "marker" in data:
                marker = data["marker"]
                ax.plot(time, data_value, linestyle, linewidth=1.7, marker=marker, markevery=3, markerfacecoloralt='none', markersize=4, color=color, label=method_name)
            else:
                ax.plot(time, data_value, linestyle, linewidth=1.4, color=color, label=method_name)

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Classification Accuracy (%)", fontsize=8)
        ax.set_xticks(time[0:13:2])
        ax.set_xticklabels(
            (
                "$\mathregular{2^{1}}$",
                "$\mathregular{2^{3}}$",
                "$\mathregular{2^{5}}$",
                "$\mathregular{2^{7}}$",
                "$\mathregular{2^{9}}$",
                "$\mathregular{2^{11}}$",
                "$\mathregular{2^{13}}$",
                # "$\mathregular{2^{17}}$",
                # "$\mathregular{2^{21}}$",
                # "$\mathregular{2^{25}}$",
                )
        )
        # ax.legend(loc=0, prop={"size": 8})
        plt.tight_layout()
        if save_fig and save_dir is not None:
            fig.savefig(save_dir + file_name + ".pdf", dpi=300)
        os.system(f"zathura {save_dir}{file_name}.pdf")
        # plt.show()

######################################################################
#                                                                    #
#               Error10 vs Error 11                                  #
#                                                                    #
######################################################################
# # network = "quantized-inception"
# error10 = pd.read_csv("./results/resnet18-cifar10-error10-only.csv")
# error11 = pd.read_csv("./results/error11-fixed-code.csv")

# error10_dict = {"Method name": "Error10", "data": error10["Acc."].to_numpy(), "style":"-", "color":"black"}
# error11_dict = {"Method name": "Error11", "data": error11["Acc."].to_numpy(), "style":"--", "color":"black"}

# # proposed_method1_dict = {
# #     "Method name": "Err",
# #     "data": proposed_method1["Acc."].to_numpy(),
# #     "style":"--",
# #     "marker":"+",
# #     "color":"#fdae61"
# # }
# # proposed_method2_dict = {
# #     "Method name": "Hydra <26-6>",
# #     "data": proposed_method2["Acc."].to_numpy(),
# #     "style":"--",
# #     "color":"#d7191c"
# # }

# data_list = (error10_dict, error11_dict)
# time = np.arange(25)

# plot_graph(
#     time,
#     data_list,
#     save_fig=True,
#     file_name="drift-error-impact",
#     save_dir="./Figures/",
# )

######################################################################
#                                                                    #
#               Error10 and Error 11 matissa only - vs normal        #
#                                                                    #
######################################################################
# proposed_method = pd.read_csv("./results/proposed_method_1_year.csv")
# baseline = pd.read_csv("./results/error_mlc_1_year.csv")

# proposed_method = {
#     "Method name": "Tri-level + 4 levels",
#     "data": proposed_method["Acc."].to_numpy(),
# }
# baseline = {"Method name": "4lv cells weights", "data": baseline["Acc."].to_numpy()}

# data_list = (proposed_method, baseline)
# time = np.arange(25)

# plot_graph_1year(
#     time,
#     data_list,
#     save_fig=True,
#     file_name="proposed_method_1_year",
#     save_dir="./Figures/",
# )

######################################################################
#                                                                    #
#               Plot final results                                   #
#                                                                    #
######################################################################
# network = "quantized-inception"
# baseline = pd.read_csv(f"./results/{network}-cifar10-baseline.csv")
# proposed_method1 = pd.read_csv(f"./results/{network}-cifar10-proposed-12-4.csv")
# proposed_method2 = pd.read_csv(f"./results/{network}-cifar10-proposed-10-6.csv")
# flipcy = pd.read_csv(f"./results/{network}-cifar10-flipcy-16bsize.csv")
# helmet = pd.read_csv(f"./results/{network}-cifar10-helmet-16bsize.csv")

# baseline_dict = {"Method name": "Baseline", "data": baseline["Acc."].to_numpy(), "style":"-", "color":"#2c7bb6"}
# flipcy_dict = {"Method name": "Flipcy", "data": flipcy["Acc."].to_numpy(), "style":":", "color":"#2c7bb6"}
# helmet_dict = {"Method name": "Helmet", "data": helmet["Acc."].to_numpy(), "style":":", "marker":"o", "color":"#fee090"}

# proposed_method1_dict = {
#     "Method name": "Hydra <28-4>",
#     "data": proposed_method1["Acc."].to_numpy(),
#     "style":"--",
#     "marker":"+",
#     "color":"#fdae61"
# }
# proposed_method2_dict = {
#     "Method name": "Hydra <26-6>",
#     "data": proposed_method2["Acc."].to_numpy(),
#     "style":"--",
#     "color":"#d7191c"
# }

# data_list = (flipcy_dict, helmet_dict, baseline_dict, proposed_method1_dict, proposed_method2_dict)
# time = np.arange(25)

# plot_graph_1year(
#     time,
#     data_list,
#     save_fig=True,
#     file_name=f"{network}-cifar10-16bsize",
#     save_dir="./Figures/",
# )
######################################################################
#                                                                    #
#               Plot different test cases                            #
#                                                                    #
######################################################################

# list_case = ("test-cases-1-10-00-01",
#                 "test-cases-2-11-00-01",
#                 "test-cases-3-01-11-10",
#                 "test-cases-4-00-11-10",
#                 "test-cases-5-01-11-00",
#                 "test-cases-6-10-11-00",
#                 "test-cases-7-00-11-01",
#                 "test-cases-8-10-11-01",)

# data_list = []
# for case in list_case:
#     data = pd.read_csv(f"./result/{case}.csv")["Acc."].to_numpy()
#     name = case
#     case_dict = {"Method name": name, "data": data}
#     data_list.append(case_dict)
# time = np.arange(25)

# plot_graph_1year(
#     time,
#     data_list,
#     save_fig=True,
#     file_name="test_case2_withoutlg",
#     save_dir="./Figures/",
# )
# baseline = pd.read_csv("./result/Pattern-00-to-11.csv")
# proposed_resnet = pd.read_csv("./result/resnet-cifar10-proposed.csv")
# proposed_lenet = pd.read_csv("./result/lenet-cifar10-proposed.csv")
# proposed_inception = pd.read_csv("./result/inception-cifar10-proposed.csv")

# baseline_dict = {"Method name": "Baseline", "data": baseline["Acc."].to_numpy(), "color":"#2c7bb6"}
# resnet_dict = {"Method name": "Resnet18", "data": proposed_resnet["Acc."].to_numpy(),  "color":"#2c7bb6"}
# googlenet_dict = {"Method name": "GoogleNet", "data": proposed_lenet["Acc."].to_numpy(),  "color":"#fee090"}
# inception_dict = {"Method name": "GoogleNet", "data": proposed_inception["Acc."].to_numpy()}

# data_list = (baseline_dict, resnet_dict, googlenet_dict, inception_dict)
# time = np.arange(25)

# plot_graph_1year(
#     time,
#     data_list,
#     save_fig=True,
#     file_name="proposed_encoding",
#     save_dir="./Figures/",
# )

######################################################################
#                                                                    #
#               Plot resnet result only #
#                                                                    #
######################################################################
# network = "quantized-inception"
# baseline = pd.read_csv("./result/resnet18-baseline.csv")

# baseline_dict = {"Method name": "Baseline", "data": baseline["Acc."].to_numpy(), "color":"black"}

# time = np.arange(25)

# plot_graph(
#     time,
#     baseline_dict,
#     save_fig=True,
#     file_name="resnet18-baseline-only",
#     save_dir="./Figures/",
# )
######################################################################
#                                                                    #
#               Plot proposed encoding                               #
#                                                                    #
######################################################################

# baseline = pd.read_csv("./result/Pattern-00-to-11.csv")
# proposed_resnet = pd.read_csv("./result/resnet-cifar10-proposed.csv")
# proposed_lenet = pd.read_csv("./result/lenet-cifar10-proposed.csv")
# proposed_inception = pd.read_csv("./result/inception-cifar10-proposed.csv")

# baseline_dict = {"Method name": "Baseline", "data": baseline["Acc."].to_numpy(), "color":"#2c7bb6"}
# resnet_dict = {"Method name": "Resnet18", "data": proposed_resnet["Acc."].to_numpy(),  "color":"#2c7bb6"}
# googlenet_dict = {"Method name": "GoogleNet", "data": proposed_lenet["Acc."].to_numpy(),  "color":"#fee090"}
# inception_dict = {"Method name": "GoogleNet", "data": proposed_inception["Acc."].to_numpy()}

# data_list = (baseline_dict, resnet_dict, googlenet_dict, inception_dict)
# time = np.arange(25)

# plot_graph_1year(
#     time,
#     data_list,
#     save_fig=True,
#     file_name="proposed_encoding",
#     save_dir="./Figures/",
# )
######################################################################
#                                                                    #
#               Plot final results imagenet                           #
#                                                                    #
######################################################################
# network = "Inception_v3"
# # baseline = pd.read_csv(f"./result/{network}-baseline.csv")
# # proposed_method = pd.read_csv(f"./result/{network}-grayencode.csv")

# baseline = pd.read_csv(f"./imagenet_results/{network}-imagenet-baseline.csv")
# proposed_method = pd.read_csv(f"./imagenet_results/{network}-imagenet-grayencode.csv")
# flipcy = pd.read_csv(f"./imagenet_results/{network}-imagenet-flipcy.csv")
# helmet = pd.read_csv(f"./imagenet_results/{network}-imagenet-helmet.csv")

# baseline_dict = {"Method name": "Baseline", "data": baseline["Acc."].to_numpy()[0:13], "style":"-", "color":"black"}
# flipcy_dict = {"Method name": "Baseline", "data": flipcy["Acc."].to_numpy()[0:13], "style":"-", "color":"#bdbdbd"}
# helmet_dict = {"Method name": "Baseline", "data": helmet["Acc."].to_numpy()[0:13], "style":":", "color":"#636363"}

# proposed_method_dict = {
#     "Method name": "Aspen",
#     "data": proposed_method["Acc."].to_numpy()[0:13],
#     "style":"--",
#     "color":"black"
# }

# data_list = (baseline_dict, flipcy_dict, helmet_dict, proposed_method_dict)
# time = np.arange(13)

# plot_graph_1year_imagenet(
#     time,
#     data_list,
#     save_fig=True,
#     file_name=f"{network}-imagenet-result-final",
#     save_dir="./Figures/",
# )
######################################################################
#                                                                    #
#               Plot final results cifar10                           #
#                                                                    #
######################################################################
network = "resnet18"
baseline = pd.read_csv(f"./results-2022/{network}-cifar10-baseline-2022.csv")
proposed_method = pd.read_csv(f"./results-2022/{network}-cifar10-proposed_method-2022.csv")
flipcy = pd.read_csv(f"././results-2022/{network}-cifar10-flipcy-2022.csv")
helmet = pd.read_csv(f"./results-2022/{network}-cifar10-helmet-2022.csv")

baseline_dict = {"Method name": "Baseline", "data": baseline["Acc."].to_numpy(), "style":"-", "color":"black"}
flipcy_dict = {"Method name": "Flipcy", "data": flipcy["Acc."].to_numpy(), "style":"-", "color":"#bdbdbd"}
helmet_dict = {"Method name": "Helmet", "data": helmet["Acc."].to_numpy(), "style":":", "color":"#636363"}

proposed_method_dict = {
    "Method name": "Aspen",
    "data": proposed_method["Acc."].to_numpy(),
    "style":"--",
    "color":"black"
}

data_list = (baseline_dict, flipcy_dict, proposed_method_dict, helmet_dict)
time = np.arange(25)

plot_graph_1year(
    time,
    data_list,
    save_fig=True,
    file_name=f"{network}-cifar10-result-method3",
    save_dir="./Figures/",
)

######################################################################
#                                                                    #
#               Plot bits sensitivity                                #
#                                                                    #
######################################################################
# with plt.style.context(['science', 'no-latex']):
#     fig, ax = plt.subplots(figsize=(3.5, 3.5))
#     data = pd.read_csv("./results/resnet18-bit_pos_sens.csv", header=None)
#     data = data.iloc[:, 2]
#     data2 = pd.read_csv("./results/LeNet-bit_pos_sens.csv", header=None)
#     data2 = data2.iloc[:, 2]
#     acc_loss = 90.690 - data
#     acc_loss2 = 92.730 - data2
#     bit_pos = np.arange(32)[::-1]
#     width = 0.5

#     ax.bar(bit_pos[:15] + width/2, acc_loss[:15], width, color="black", label="Resnet18 on CIFAR10")
#     ax.bar(bit_pos[:15] - width/2, acc_loss2[:15], width, hatch="/", fill=False, visible=True, label="LeNet on CIFAR10")

#     ax.set_xlabel('Bit position', fontsize=8)
#     ax.set_ylabel('Accuracy Loss (%)', fontsize=8)
#     ax.invert_xaxis()
#     ax.legend(loc=0, prop={'size': 8})
#     plt.show()
#     fig.savefig("./Figures/Bit_sens2.svg",  dpi=300)

######################################################################
#                                                                    #
#               Plot layer sensitivity                               #
#                                                                    #
######################################################################

# data = pd.read_csv("./results/layer_sens2.csv")
# layer_name = ("conv1.weight",
#               "layer1.0.conv1.weight",
#               "layer1.0.conv2.weight",
#               "layer1.1.conv1.weight",
#               "layer1.1.conv2.weight",
#               "layer2.0.conv1.weight",
#               "layer2.0.conv2.weight",
#               # "layer2.0.shortcut.0.weight",
#               # "layer2.0.shortcut.1.weight",
#               "layer2.1.conv1.weight",
#               "layer2.1.conv2.weight",
#               "layer3.0.conv1.weight",
#               "layer3.0.conv2.weight",
#               # "layer3.0.shortcut.0.weight",
#               # "layer3.0.shortcut.1.weight",
#               "layer3.1.conv1.weight",
#               "layer3.1.conv2.weight",
#               "layer4.0.conv1.weight",
#               "layer4.0.conv2.weight",
#               # "layer4.0.shortcut.0.weight",
#               # "layer4.0.shortcut.1.weight",
#               "layer4.1.conv1.weight",
#               "layer4.1.conv2.weight",
#               "linear.weight")

# list_acc = []

# with plt.style.context(['science', 'no-latex']):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     for index, name in enumerate(layer_name):
#         data_layer = data[data["Layer"] == name]
#         acc = data_layer["Acc."]
#         time = np.arange(14)
#         if index <=6:
#             ax.plot(time, acc, label=name)
#         elif index <= 13:
#             ax.plot(time, acc, marker='o', label=name)
#         else:
#             ax.plot(time, acc, marker='v', label=name)

#     ax.set_xlabel('Time (s)', fontsize=8)
#     ax.set_ylabel('Cifa10 Test Accuracy (%)', fontsize=8)
#     ax.set_xticks(time)
#     ax.set_xticklabels(("$2^{1}$", "$2^{2}$","$2^{3}$","$2^{4}$","$2^{5}$",
#             "$2^{6}$","$2^{7}$","$2^{8}$","$2^{9}$","$2^{10}$","$2^{11}$","$2^{12}$","$2^{13}$","$2^{14}$"))
#     ax.legend(loc=0, prop={'size': 8})
#     plt.show()
#     fig.savefig("Layer_sens.pdf",  dpi=300)

######################################################################
#                                                                    #
#               Layer40 and Layer30 MLC with clipping                #
#                                                                    #
######################################################################
# layer40_data = pd.read_csv("./results/layer40_conv1_sens.csv")
# layer30_data = pd.read_csv("./results/layer30_conv1_sens.csv")
# layer40 = {"Method name": "Layer4.0.conv1.weight",
#                 "data": layer40_data["Acc."].to_numpy()
#                 }

# layer30 = {"Method name": "Layer3.0.conv1.weight",
#                 "data": layer30_data["Acc."].to_numpy()
#                 }

# data_list = (layer40, layer30)
# time = np.arange(14)

# # plot_graph(time, data_list, save_fig=True, save_dir="./Figures/")
# plot_graph(time, data_list, save_fig=True, file_name="layer_clipping", save_dir="./Figures/")
