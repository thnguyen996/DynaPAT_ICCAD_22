"""Count the number pattern 11, 10, 01, and 00"""
from cifar10_models import *
from functools import partial
from matplotlib import font_manager
from matplotlib import rc
from matplotlib import rcParams
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import progress_bar
from weight_quantized_conf import *
import argparse
import copy
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import traceback
torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--mlc", default=8, type=int, help="Number of mlc bits")
parser.add_argument("--name", default="Name", type=str, help="Name of run")
parser.add_argument("--method", default="proposed_method", type=str, help="Running method")
parser.add_argument("--model", default="resnet18", type=str, help="Model")
parser.add_argument("--gpu", default="0", type=str, help="GPU ids")
parser.add_argument(
    "--save_data", "-s", action="store_true", help="Save the data")
parser.add_argument(
    "--encode", "-e", action="store_true", help="Encoding for flipcy and helmet")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument("--num_bits", default=8, type=int, help="Number of quantized bits")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="~/Datasets/cifar10/", train=True, download=False, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=500, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="~/Datasets/cifar10/", train=False, download=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    print("==> Building model..")
    if args.model == "resnet18":
        net = ResNet18()
        net.load_state_dict(torch.load("./checkpoint/resnet.pt")['net'])
    elif args.model == "LeNet":
        net = googlenet()
        net.load_state_dict(torch.load("./checkpoint/googlenet.pt"))
    elif args.model == "Inception":
        net = Inception3()
        net.load_state_dict(torch.load("./checkpoint/inception_v3.pt"))

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Quantize the model

    with torch.no_grad():
        for name, weight in net.named_parameters():
            if ( "weight" in name ) and ( "bn" not in name ) and ("shortcut" not in name):
                qi, qf = (3, 13)
                (imin, imax) = (-np.exp2(qi-1), np.exp2(qi-1)-1)
                fdiv = np.exp2(-qf)
                quantized_weight = torch.mul(torch.round(torch.div(weight, fdiv)), fdiv)
                quantized_weight = torch.clamp(quantized_weight, min=imin, max=imax)
                weight.copy_(quantized_weight)

    # orig_state_dict = copy.deepcopy(net.state_dict())
    acc = test(net, criterion, optimizer, testloader, device)
    print(acc)
    # with torch.no_grad():
    #     list_11 = [3]
    #     list_10 = [2]
    #     list_01 = [1]
    #     list_00 = [0]
    #     for shift in range(2, args.num_bits, 2):
    #         next_pos = 2 ** (shift)
    #         list_11.append(3 * next_pos)
    #         list_10.append(2 * next_pos)
    #         list_01.append(1 * next_pos)
    #         list_00.append(0)
    #     tensor_11 = np.array(list_11, dtype=dtype)
    #     tensor_10 = np.array(list_10, dtype=dtype)
    #     tensor_01 = np.array(list_01, dtype=dtype)
    #     tensor_00 = np.array(list_00, dtype=dtype)
    #     index_bit = np.arange(0, args.num_bits, 2)

    #     # total11 = np.array([0, 0, 0, 0])
    #     # total10 = np.array([0, 0, 0, 0])
    #     # total01 = np.array([0, 0, 0, 0])
    #     # total00 = np.array([0, 0, 0, 0])
    #     total11 = []
    #     total10 = []
    #     total01 = []
    #     total00 = []

    #     index = 1

    #     for (name, weight) in tqdm(net.named_parameters(), desc="Counting pattern: ", leave=False):
    #         if ( "weight" in name ) and ( "bn" not in name ) and ("shortcut" not in name):
    #             weight = weight.cpu().numpy().astype(dtype)
    #             _, index_11 = count(weight, tensor_11, tensor_11, index_bit, num_bits=args.num_bits)
    #             _, index_01 = count(weight, tensor_01, tensor_11, index_bit, num_bits=args.num_bits)
    #             _, index_10 = count(weight, tensor_10, tensor_11, index_bit, num_bits=args.num_bits)
    #             _, index_00 = count(weight, tensor_00, tensor_11, index_bit, num_bits=args.num_bits)

    #             num_11 = np.unique(index_11[:, 1], return_counts=True)[1]
    #             num_01 = np.unique(index_01[:, 1], return_counts=True)[1]
    #             num_10 = np.unique(index_10[:, 1], return_counts=True)[1]
    #             num_00 = np.unique(index_00[:, 1], return_counts=True)[1]

    #             # total11+=num_11
    #             # total01+=num_01
    #             # total10+=num_10
    #             # total00+=num_00
    #             total = num_11 + num_01 + num_10 + num_00
    #             num_11 = num_11/total*100
    #             num_10 = num_10/total*100
    #             num_01 = num_01/total*100
    #             num_00 = num_00/total*100
    #             total11.append(num_11)
    #             total10.append(num_10)
    #             total01.append(num_01)
    #             total00.append(num_00)

    # total11 = np.array(total11)
    # total01 = np.array(total01)
    # total10 = np.array(total10)
    # total00 = np.array(total00)

    # with plt.style.context(['ieee', 'no-latex']):
    #     mpl.rcParams['font.family'] = 'NimbusRomNo9L'
    #     fig, ax = plt.subplots(figsize=(6, 2))
    #     labels = (np.arange(4, dtype=np.float64) + 1)*10
    #     width = 0.5      # the width of the bars: can also be len(x) sequence
    #     total1110 = total11 + total10
    #     total111001 = total11 + total10 + total01

    #     ax.bar(labels, total11[0, :], width, edgecolor="black", color="#d7191c", label='11', align='center')
    #     ax.bar(labels, total10[0, :], width, edgecolor="black", color="#fdae61", bottom=total11[0, :],
    #            label='10', align='center')
    #     ax.bar(labels, total01[0, :], width, edgecolor="black", color="#abd9e9",  bottom=total1110[0, :],
    #            label='01', align='center')
    #     ax.bar(labels, total00[0, :], width, edgecolor="black", color="#2c7bb6",  bottom=total111001[0, :],
    #            label='00', align='center')

    #     for i in range(1, 18):
    #         labels += 0.5
    #         ax.bar(labels, total11[i, :], width, edgecolor="black", color="#d7191c",  align='center')
    #         ax.bar(labels, total10[i, :], width, edgecolor="black", color="#fdae61", bottom=total11[i, :],
    #                 align='center')
    #         ax.bar(labels, total01[i, :], width, edgecolor="black", color="#abd9e9",  bottom=total1110[i, :],
    #                 align='center')
    #         ax.bar(labels, total00[i, :], width, edgecolor="black", color="#2c7bb6",  bottom=total111001[i, :],
    #                 align='center')
    #     # ax.legend(loc=0, prop={"size": 14})
    # ax.set_xlabel("Time (s)", fontsize=8)
    # ax.set_ylabel("Cifa10 Test Accuracy (%)", fontsize=8)
    # plt.tight_layout()
    # fig.savefig("./Figures/Inception_count_pattern_quantized_layers_error.pdf", dpi=300)
    # # plt.show()
    # os.system("zathura ./Figures/Inception_count_pattern_quantized_layers_error.pdf")

def count(weight, tensor_10, tensor_11, index_bit, num_bits):
    num_10 = 0
    indices_10 = []
    weight = weight.flatten()
    for tensor_10_i, tensor_11_i, index_b in zip(tensor_10, tensor_11, index_bit):
        and_result = np.bitwise_and(tensor_11_i, weight)
        index_10 = (and_result == tensor_10_i).nonzero()[0]
        bit_index = np.full_like(index_10, index_b)
        index_tensor = np.stack((index_10, bit_index))
        indices_10.append(index_tensor)
        num_10 += index_10.size
    total_index_10 = np.concatenate(indices_10, axis=1)
    total_index_10 = np.transpose(total_index_10, (1, 0))
    return num_10, total_index_10

def test(net, criterion, optimizer, testloader, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
    acc = 100.0 * correct / total
    return acc

if __name__ == "__main__":
    main()
