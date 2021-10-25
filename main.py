"""Evaluate on cifar10 32-bit floating point"""

from cifar10_models import *
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from utils import progress_bar
from weight_cupy_conf import *
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
from flipcy import *
from helmet import *
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
    iteration = 100
    weight_type = {"MLC": args.mlc, "SLC": 32-args.mlc}

# Config weight and inject error:
    tensorboard = args.save_data
    save_data = args.save_data
    if save_data:
        df = pd.DataFrame( columns= ['Time', 'Acc.'])

    if tensorboard:
        ran = random.randint(1, 100)
        writer = SummaryWriter(f"./runs/{args.name}-{ran}")
        print(f"Run ID: {ran}")
    with torch.no_grad():
        count = 0
        for (name1, value1) , (name2, value2) in zip(error_11_pd.iteritems(), error_10_pd.iteritems()):
            error_11 = value1[0]
            error_10 = None
            running_acc = []
            print("Evaluating for error_rate of level 2: {}".format(error_10))
            print("Evaluating for error_rate of level 3: {}".format(error_11))
            if args.encode & (count > 0):
                print("Done Encoding...")
                break
            for it in range(iteration):
                if args.model == "resnet18":
                    state_dict = torch.load(state_dict_path)["net"]
                else:
                    state_dict = torch.load(state_dict_path)
                if args.encode & (it > 0):
                    break
                for (name, weight) in tqdm(state_dict.items(), desc="Executing method:", leave=False):
                    if ( "weight" in name ) and ( "bn" not in name ):
                        mlc_error_rate = {"error_11" : error_11, "error_10": error_10}
                        if args.method == "proposed_method":
                            error_weight = proposed_method(weight, weight_type, mlc_error_rate)
                        if args.method == "flipcy":
                            error_weight = flipcy(weight, weight_type, mlc_error_rate, name, encode=args.encode)
                        if args.method == "helmet":
                            error_weight = helmet(weight, weight_type, mlc_error_rate, name, encode=args.encode)
                        weight.copy_(error_weight)
                net.load_state_dict(state_dict)
                class_acc = test(net, criterion, optimizer, testloader, device)
                running_acc.append(class_acc)
            avr_acc = sum(running_acc) / len(running_acc)
            if tensorboard and not args.encode:
                writer.add_scalar("Acc./", avr_acc, count)
                writer.close()
            if save_data and not args.encode:
                df.loc[count] = [count, avr_acc]
                df.to_csv(f"./results/{args.name}.csv", mode='w', header=True)
            count += 1

def proposed_method(weight, weight_type, mlc_error_rate):
    MLC = weight_conf(weight, weight_type)
    error_weight = MLC.inject_error(mlc_error_rate)
    error_weight = error_weight.reshape(weight.shape)
    return error_weight

def flipcy(weight, weight_type, mlc_error_rate, name, encode):
    shape = weight.shape
    MLC = weight_conf(weight, weight_type)
    if encode:
        encoded_weight, flip, comp = flipcy_en(MLC.weight)
        if not os.path.isdir(f"./flipcy_en/{args.model}"):
            os.mkdir(f"./flipcy_en/{args.model}")
        torch.save(encoded_weight, f"./flipcy_en/{args.model}/{name}.pt")
        torch.save(flip, f"./flipcy_en/{args.model}/flip-{name}.pt")
        torch.save(comp, f"./flipcy_en/{args.model}/comp-{name}.pt")
        float_weight = encoded_weight.view(cp.float32)
        weight_torch = torch.tensor(float_weight, device=weight.device).reshape(shape)
    else:
        assert os.path.isdir(f"./flipcy_en/{args.model}"), "You need to do encoding first"
        encoded_weight = torch.load(f"./flipcy_en/{args.model}/{name}.pt")
        flip = torch.load(f"./flipcy_en/{args.model}/flip-{name}.pt")
        comp = torch.load(f"./flipcy_en/{args.model}/comp-{name}.pt")
        error_weight = inject_error(weight_type, encoded_weight, mlc_error_rate)
        decoded_weight = flipcy_de(error_weight, flip, comp)
        float_weight = decoded_weight.view(cp.float32)
        weight_torch = torch.tensor(float_weight, device=weight.device).reshape(shape)
    return weight_torch

def helmet(weight, weight_type, mlc_error_rate, name, encode):
    shape = weight.shape
    MLC = weight_conf(weight, weight_type)
    if encode:
        encoded_weight, fc = helmet_en(MLC.weight)
        if not os.path.isdir(f"./helmet_en/{args.model}"):
            os.mkdir(f"./helmet_en/{args.model}")
        torch.save(encoded_weight, f"./helmet_en/{args.model}/{name}.pt")
        torch.save(fc, f"./helmet_en/{args.model}/fc-{name}.pt")
        float_weight = encoded_weight.view(cp.float32)
        weight_torch = torch.tensor(float_weight, device=weight.device).reshape(shape)
    else:
        assert os.path.isdir(f"./helmet_en/{args.model}"), "You need to do encoding first"
        encoded_weight = torch.load(f"./helmet_en/{args.model}/{name}.pt")
        fc = torch.load(f"./helmet_en/{args.model}/fc-{name}.pt")
        error_weight = helmet_inject_error(weight_type, encoded_weight, mlc_error_rate)
        decoded_weight = helmet_de(error_weight, fc)
        float_weight = decoded_weight.view(cp.float32)
        weight_torch = torch.tensor(float_weight, device=weight.device).reshape(shape)
    return weight_torch

# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


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
# if args.method == "method0":
#     SAsimulate = makeSA.sa_config(
#         testloader,
#         net,
#         state_dict,
#         args.method,
#         writer=False
#     )
#     error_range = np.logspace(-2, -1, 100)
#     if not os.path.isdir("./save_cp"):
#         os.mkdir("./save_cp")
#         SAsimulate.np_to_cp()
#     SAsimulate.run(error_range, 100, test, 0, state_dict, "./save_cp/")

