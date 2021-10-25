"""Evaluate on cifar10 32-bit floating point"""

from cifar10_models import *
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from utils import progress_bar
from weight_pattern_sens import *
import argparse
import numpy as np
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
from tqdm import tqdm
# from flipcy import flipcy_en, inject_error
# from helmet import helmet_en
torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--mlc", default=32, type=int, help="Number of mlc bits")
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

# Model
    print("==> Building model..")
    if args.model == "resnet18":
        net = ResNet18()
        state_dict_path = ("./checkpoint/resnet.pt")
    elif args.model == "LeNet":
        net = googlenet()
        state_dict_path = ("./checkpoint/googlenet.pt")
    elif args.model == "Inception":
        net = Inception3()
        state_dict_path = ("./checkpoint/inception_v3.pt")
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # net.load_state_dict(torch.load(state_dict_path))

    # class_acc = test(net, criterion, optimizer, testloader, device)

# Load error rate:
    error_11_pd = pd.read_csv("./error_level3_1year.csv", header = None, delimiter="\t")
    error_10_pd = pd.read_csv("./error_level2_1year.csv", header = None, delimiter="\t")
    error_10_pd = error_10_pd
    error_11_pd = error_11_pd
    iteration = 100

# Config weight and inject error:
    tensorboard = args.save_data
    save_data = args.save_data
    error_pats = ["00", "01",  "10", "11"]
    des_pats = ["00", "01",  "10", "11"]


    with torch.no_grad():
        for error_pat in error_pats:
            for des_pat in des_pats:
                count = 0
                if des_pat == error_pat:
                    continue
                else:
                    if save_data:
                        df = pd.DataFrame( columns= ['Time', 'error_pat', 'des_pat', 'Acc.'])
                        # ran = random.randint(1, 100)
                        writer = SummaryWriter(f"./runs/LeNet-pattern-{error_pat}-to-{des_pat}")
                        # print(f"Run ID: {ran}")
                    print(f"Running experiment for Error: {error_pat} --> {des_pat}")
                    for (name, value) in error_11_pd.iteritems():
                        print(f"Running experiment for error rate: {value}")
                        error = value[0]
                        running_acc = []
                        for it in range(iteration):
                            if args.model == "resnet18":
                                state_dict = torch.load(state_dict_path)["net"]
                            else:
                                state_dict = torch.load(state_dict_path)
                            if args.encode & (it > 0):
                                break
                            for (name, weight) in tqdm(state_dict.items(), desc="Executing method:", leave=False):
                                if ( "weight" in name ) and ( "bn" not in name ):
                                    mlc_error_rate = {"error_rate" : error}
                                    if args.method == "proposed_method":
                                        error_weight = proposed_method(weight, error_pat, des_pat, mlc_error_rate)
                                    weight.copy_(error_weight)
                            net.load_state_dict(state_dict)
                            class_acc = test(net, criterion, optimizer, testloader, device)
                            running_acc.append(class_acc)
                        avr_acc = sum(running_acc) / len(running_acc)
                        if tensorboard:
                            writer.add_scalar("Acc./", avr_acc, count)
                            writer.close()
                        if save_data:
                            df.loc[count] = [count, error_pat, des_pat, avr_acc]
                            df.to_csv(f"./result/LeNet-Pattern-{error_pat}-to-{des_pat}.csv", mode='w', header=True)
                        count += 1

def proposed_method(weight, error_pat, des_pat, mlc_error_rate):
    MLC = weight_conf(weight)
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
