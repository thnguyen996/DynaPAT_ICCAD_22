"""Evaluate on cifar10 32-bit floating point"""

from cifar10_models import *
from distiller.data_loggers.collector import collector_context
from functools import partial
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from utils import progress_bar
from weight_pattern_sens_quantized import *
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

torch.set_printoptions(profile="full")
msglogger = logging.getLogger()

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--mlc", default=8, type=int, help="Number of mlc bits")
parser.add_argument("--name", default="Name", type=str, help="Name of run")
parser.add_argument(
    "--method", default="proposed_method", type=str, help="Running method"
)
parser.add_argument("--model", default="resnet18", type=str, help="Model")
parser.add_argument("--gpu", default="0", type=str, help="GPU ids")
parser.add_argument("--save_data", "-s", action="store_true", help="Save the data")
parser.add_argument(
    "--encode", "-e", action="store_true", help="Encoding for flipcy and helmet"
)
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
        root="~/Datasets/cifar10/",
        train=True,
        download=False,
        transform=transform_train,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=500, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="~/Datasets/cifar10/",
        train=False,
        download=False,
        transform=transform_test,
    )
    _, val_set = torch.utils.data.random_split(testset, [9500, 500])

    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=100, shuffle=False, num_workers=2
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

    # Model
    print("==> Building model..")
    if args.model == "resnet18":
        net = ResNet18()
        net.load_state_dict(torch.load("./checkpoint/resnet.pt")["net"])
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
        msglogger.info(
            "Generating quantization calibration stats based on {0} users".format(
                args.qe_calibration
            )
        )
        collector = distiller.data_loggers.QuantCalibrationStatsCollector(net)
        with collector_context(collector):
            test(net, criterion, optimizer, valloader, device="cuda")
        yaml_path = f"./quant_stats/{args.model}.yaml"
        collector.save(yaml_path)
        print("Done")

    # Quantize the model
    assert os.path.isfile(
        f"./quant_stats/{args.model}.yaml"
    ), "You should do activation calibration first"
    args.qe_stats_file = f"./quant_stats/{args.model}.yaml"
    args.qe_bits_acts = args.num_bits
    args.qe_bits_wts = args.num_bits
    quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(net, args)
    quantizer.prepare_model(torch.randn(100, 3, 32, 32))
    orig_state_dict = copy.deepcopy(net.state_dict())

    # Load error rate:
    error_11_pd = pd.read_csv("./error_level3_1year.csv", header=None, delimiter="\t")
    error_10_pd = pd.read_csv("./error_level2_1year.csv", header=None, delimiter="\t")
    error_10_pd = error_10_pd
    error_11_pd = error_11_pd
    iteration = 100

    # # Config weight and inject error:
    error_pats = ["00", "01", "10", "11"]
    des_pats = ["00", "01", "10", "11"]

    with torch.no_grad():
        for error_pat in error_pats:
            for des_pat in des_pats:
                count = 0
                if des_pat == error_pat:
                    continue
                else:
                    if args.save_data:
                        df = pd.DataFrame(
                            columns=["Time", "error_pat", "des_pat", "Acc."]
                        )
                        # ran = random.randint(1, 100)
                        writer = SummaryWriter(
                            f"./runs/quantized-{args.num_bits}bit-{args.model}-pattern-{error_pat}-to-{des_pat}"
                        )
                        # print(f"Run ID: {ran}")
                    print(
                        f"Running experiment for quantized-{args.model}-{args.num_bits}bit Error: {error_pat} --> {des_pat}"
                    )
                    for (name, value) in error_11_pd.iteritems():
                        print(f"Running experiment for error rate: {value}")
                        error = value[0]
                        running_acc = []
                        for it in range(iteration):
                            # Reset parameters:
                            net.load_state_dict(orig_state_dict)
                            for (name, weight) in tqdm(
                                net.named_parameters(),
                                desc="Executing method:",
                                leave=False,
                            ):
                                if ("weight" in name) and ("bn" not in name):
                                    mlc_error_rate = {"error_rate": error}
                                    if args.method == "proposed_method":
                                        error_weight = proposed_method(
                                            weight,
                                            error_pat,
                                            des_pat,
                                            mlc_error_rate,
                                            args.num_bits,
                                        )
                                    weight.copy_(error_weight)
                            class_acc = test(
                                net, criterion, optimizer, testloader, device
                            )
                            running_acc.append(class_acc)
                        avr_acc = sum(running_acc) / len(running_acc)
                        if args.save_data:
                            writer.add_scalar("Acc./", avr_acc, count)
                            writer.close()
                            df.loc[count] = [count, error_pat, des_pat, avr_acc]
                            df.to_csv(
                                f"./result/quantized-{args.num_bits}bit-{args.model}-Pattern-{error_pat}-to-{des_pat}.csv",
                                mode="w",
                                header=True,
                            )
                        count += 1

def proposed_method(weight, error_pat, des_pat, mlc_error_rate, num_bits):
    MLC = weight_conf(weight, num_bits)
    error_weight = MLC.inject_error(mlc_error_rate, error_pat, des_pat)
    error_weight = error_weight.reshape(weight.shape)
    return error_weight


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
