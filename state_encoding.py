"""Count the number pattern 11, 10, 01, and 00"""
import matplotlib.pyplot as plt
from cifar10_models import *
from distiller.data_loggers.collector import collector_context
from functools import partial
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import progress_bar
import argparse
import copy
import distiller
import logging
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
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib import rc
import matplotlib as mpl
from weight_quantized_conf import *
torch.set_printoptions(profile="full")

msglogger = logging.getLogger()
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
distiller.quantization.add_post_train_quant_args(parser)
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

    if not os.path.isfile(f"./quant_stats/{args.model}.yaml"):
        distiller.utils.assign_layer_fq_names(net)
        msglogger.info("Generating quantization calibration stats based on {0} users".format(args.qe_calibration))
        collector = distiller.data_loggers.QuantCalibrationStatsCollector(net)
        with collector_context(collector):
            test(net, criterion, optimizer, valloader, device="cuda")
        yaml_path = f"./quant_stats/{args.model}.yaml"
        collector.save(yaml_path)
        print("Done")

    # Quantize the model
    assert os.path.isfile(f"./quant_stats/{args.model}.yaml"), "You should do activation calibration first"
    args.qe_stats_file = f"./quant_stats/{args.model}.yaml"
    args.qe_bits_acts = args.num_bits
    args.qe_bits_wts = args.num_bits
    args.qe_config_file = "./resnet18_imagenet_post_train.yaml"
    quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(net, args)
    quantizer.prepare_model(torch.randn(100, 3, 32, 32))
    orig_state_dict = copy.deepcopy(net.state_dict())

# Count number of pattern:
    if args.num_bits == 16:
        dtype = np.uint16
    elif args.num_bits == 8:
        dtype = np.uint8

    with torch.no_grad():
        list_11 = [3]
        list_10 = [2]
        list_01 = [1]
        list_00 = [0]
        for shift in range(2, args.num_bits, 2):
            next_pos = 2 ** (shift)
            list_11.append(3 * next_pos)
            list_10.append(2 * next_pos)
            list_01.append(1 * next_pos)
            list_00.append(0)
        tensor_11 = np.array(list_11, dtype=dtype)
        tensor_10 = np.array(list_10, dtype=dtype)
        tensor_01 = np.array(list_01, dtype=dtype)
        tensor_00 = np.array(list_00, dtype=dtype)
        index_bit = np.arange(0, args.num_bits, 2)

        total11 = []
        total10 = []
        total01 = []
        total00 = []
        index = 1

        for (name, weight) in tqdm(net.named_parameters(), desc="Counting pattern: ", leave=False):
            if ( "weight" in name ) and ( "bn" not in name ) and ("shortcut" not in name):
                weight = weight.cpu().numpy().astype(dtype)
                _, index_11 = count(weight, tensor_11, tensor_11, index_bit, num_bits=args.num_bits)
                _, index_01 = count(weight, tensor_01, tensor_11, index_bit, num_bits=args.num_bits)
                _, index_10 = count(weight, tensor_10, tensor_11, index_bit, num_bits=args.num_bits)
                _, index_00 = count(weight, tensor_00, tensor_11, index_bit, num_bits=args.num_bits)

                num_11 = np.unique(index_11[:, 1], return_counts=True)[1]
                num_01 = np.unique(index_01[:, 1], return_counts=True)[1]
                num_10 = np.unique(index_10[:, 1], return_counts=True)[1]
                num_00 = np.unique(index_00[:, 1], return_counts=True)[1]

                num_11 = np.sum(num_11)
                num_10 = np.sum(num_10)
                num_01 = np.sum(num_01)
                num_00 = np.sum(num_00)
                total = num_11 + num_01 + num_10 + num_00
                num_11 = num_11/total*100
                num_10 = num_10/total*100
                num_01 = num_01/total*100
                num_00 = num_00/total*100
                total11.append(num_11)
                total10.append(num_10)
                total01.append(num_01)
                total00.append(num_00)

    total11 = np.array(total11)
    total01 = np.array(total01)
    total10 = np.array(total10)
    total00 = np.array(total00)
    total = np.stack((total00, total01, total10, total11))
    index_sort = np.argsort(total, 0)
    import pdb; pdb.set_trace()

    # Plot graph
    # with plt.style.context(['ieee', 'no-latex']):
    #     mpl.rcParams['font.family'] = 'NimbusRomNo9L'
    #     fig, ax = plt.subplots(figsize=(6, 2))
    #     labels = np.arange(18)
    #     width = 0.8
    #     total1110 = total11 + total10
    #     total111001 = total11 + total10 + total01
    #     for i in range(1, 18):
    #         ax.bar(i, total11[i], width, edgecolor="black", color="#d7191c",  align='center')
    #         ax.bar(i, total10[i], width, edgecolor="black", color="#fdae61", bottom=total11[i],
    #                 align='center')
    #         ax.bar(i, total01[i], width, edgecolor="black", color="#abd9e9",  bottom=total1110[i],
    #                 align='center')
    #         ax.bar(i, total00[i], width, edgecolor="black", color="#2c7bb6",  bottom=total111001[i],
    #                 align='center')
    # ax.set_xlabel("Time (s)", fontsize=8)
    # ax.set_ylabel("Cifa10 Test Accuracy (%)", fontsize=8)
    # ax.invert_xaxis()
    # plt.tight_layout()
    # fig.savefig(f"./Figures/{args.model}_count_pattern_without_bitpos.pdf", dpi=300)
    # os.system(f"zathura ./Figures/{args.model}_count_pattern_without_bitpos.pdf")

def proposed_method(weight, weight_type, mlc_error_rate, num_bits):
    MLC = weight_conf(weight, weight_type, num_bits)
    error_weight = MLC.inject_error(mlc_error_rate)
    error_weight = error_weight.reshape(weight.shape)
    return error_weight

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
