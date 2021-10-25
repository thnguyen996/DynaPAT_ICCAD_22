"""Test quantized model"""
import argparse
import os
import numpy as np

from cifar10_models import *
from aquantizer import *
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from utils import progress_bar
from weight_quantized_conf import *
import numpy as np
import pandas as pd
import pdb
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.quantization
import torchvision
import torchvision.transforms as transforms
from flipcy_quantized import flipcy_en, count_orig, inject_error
from helmet_quantized import helmet_en
from tqdm import tqdm
import distiller
import traceback
import logging
from functools import partial
from distiller.data_loggers.collector import collector_context
import copy

msglogger = logging.getLogger()
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--mlc", default=16, type=int, help="Number of mlc bits")
parser.add_argument("--name", default="Name", type=str, help="Name of run")
parser.add_argument("--method", default="proposed_method", type=str, help="Running method")
parser.add_argument("--model", default="resnet18", type=str, help="Model")
parser.add_argument("--gpu", default="0", type=str, help="GPU ids")
parser.add_argument(
    "--save_data", "-s", action="store_true", help="Save the data")
parser.add_argument(
    "--encode", "-e", action="store_true", help="Enable encode for flipcy and helmet")
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
    _, val_set = torch.utils.data.random_split(testset, [9500, 500])

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=100, shuffle=False, num_workers=2
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

    # Activation calibrations

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
    quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(net, args)
    quantizer.prepare_model(torch.randn(100, 3, 32, 32))
    orig_state_dict = copy.deepcopy(net.state_dict())

    error_11_pd = pd.read_csv("./error_level3_1year.csv", header = None, delimiter="\t")
    error_10_pd = pd.read_csv("./error_level2_1year.csv", header = None, delimiter="\t")
    # error_11_pd = error_11_pd.loc[0, 8:]
    # error_10_pd = error_10_pd.loc[0, 8:]

    iteration = 100
    weight_type = {"MLC": args.mlc , "SLC": args.num_bits-args.mlc}
    tensorboard = args.save_data
    save_data = args.save_data
    if save_data:
        df = pd.DataFrame( columns= ['Time', 'Acc.'])
    if tensorboard:
        ran = random.randint(1, 100)
        writer = SummaryWriter(f"./runs/{args.name}-{ran}")
        print(f"Run ID: {ran}")

    list_01 = [1]
    list_10 = [2]
    list_11 = [3]
    for shift in range(2, args.num_bits, 2):
        next_pos = 2 ** (shift)
        list_01.append(next_pos)
        list_10.append(next_pos)
        list_11.append(3 * next_pos)
    if args.num_bits == 16:
        dtype=torch.int16
    elif args.num_bits == 8:
        dtype=torch.uint8
    tensor_01 = torch.tensor(list_01, dtype=dtype, device="cuda")
    tensor_10 = torch.tensor(list_10, dtype=dtype, device="cuda")
    tensor_11 = torch.tensor(list_11, dtype=dtype, device="cuda")
    tensors = (tensor_01, tensor_10, tensor_11)

    with torch.no_grad():
        count = 0
        for (name1, value1), (name2, value2) in zip(error_11_pd.iteritems(), error_10_pd.iteritems()):
            running_acc = []
            error_11 = value1[0]
            error_10 = value2[0]
            print("Evaluating for error_rate of level 2: {}".format(error_10))
            print("Evaluating for error_rate of level 3: {}".format(error_11))
            if args.encode & (count > 0):
                print("Done Encoding...")
                break
            for it in range(iteration):
                # Reset parameters:
                net.load_state_dict(orig_state_dict)
                if args.encode & (it > 0):
                    break
                for name, weight in tqdm(net.named_parameters(), desc="Executing method:", leave=False):
                    if ( "weight" in name ) and ( "bn" not in name ):
                        # level = "bit"
                        mlc_error_rate = {"error_11" : error_11, "error_10": error_10}
                        if args.method == "proposed_method":
                            error_weight = proposed_method(weight, weight_type, mlc_error_rate, args.num_bits)
                        if args.method == "flipcy":
                            error_weight = flipcy(weight, weight_type, mlc_error_rate, name, tensors, num_bits=args.num_bits, encode=args.encode)
                        if args.method == "helmet":
                            error_weight = helmet(weight, weight_type, mlc_error_rate, name, tensors, num_bits=args.num_bits, encode=args.encode)
                        weight.copy_(error_weight)

                class_acc = test(net, criterion, optimizer, testloader, device)
                running_acc.append(class_acc)
            avr_acc = sum(running_acc) / len(running_acc)
            if tensorboard and not args.encode:
                writer.add_scalar("Acc./", avr_acc, count)
                writer.close()
            if save_data and not args.encode:
                df.loc[count] = [count, avr_acc]
                df.to_csv(f"./new_results/{args.name}.csv", mode='w', header=True)
            count += 1


def proposed_method(weight, weight_type, mlc_error_rate, num_bits):
    MLC = weight_conf(weight, weight_type, num_bits)
    error_weight = MLC.inject_error(mlc_error_rate)
    error_weight = error_weight.reshape(weight.shape)
    return error_weight

# 00, 01, 11, 10
def flipcy(weight, weight_type, mlc_error_rate, name, tensors, num_bits, encode):
    shape = weight.shape
    MLC = weight_conf(weight, weight_type, num_bits)
    tensor_01, tensor_10, tensor_11 = tensors
    tensor_11 = tensor_11.cpu().numpy()
    tensor_01 = tensor_01.cpu().numpy()
    tensor_10 = tensor_10.cpu().numpy()
    if encode:
        encoded_weight = flipcy_en(MLC.weight, num_bits)
        if not os.path.isdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b"):
            os.mkdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b")
        torch.save(encoded_weight, f"./flipcy_en/quantized-{args.model}-{num_bits}b/{name}.pt")
        weight_torch = encoded_weight.reshape(shape)
    else:
        if num_bits == 8:
            dtype = np.uint8
        elif num_bits == 16:
            dtype = np.int16
        assert os.path.isdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b"), "You need to do encoding first"
        encoded_weight = torch.load(f"./flipcy_en/quantized-{args.model}-{num_bits}b/{name}.pt")

        encoded_weight = encoded_weight.cpu().numpy().astype(dtype)
        weight = weight.view(-1).cpu().numpy().astype(dtype)
        num_11_flipcy, _ = count_orig(encoded_weight, tensor_11, tensor_11, num_bits)
        num_error_11_flipcy = int(mlc_error_rate["error_11"] * num_11_flipcy)
        num_01_flipcy, _ = count_orig(encoded_weight, tensor_01, tensor_11, num_bits)
        num_error_01_flipcy = int(mlc_error_rate["error_10"] * num_01_flipcy)

        # num_01, _  = count_orig(weight, tensor_01, tensor_11, num_bits)
        # num_10, _ = count_orig(weight, tensor_10, tensor_11, num_bits)
        num_error = (num_error_11_flipcy, num_error_01_flipcy)
        error_weight = inject_error(weight, num_error, mlc_error_rate, num_bits)
        weight_torch = error_weight.reshape(shape)
        # print("Number of error 11:", num_error_11_flipcy, num_11*mlc_error_rate["error_11"])
        # print("Number of error 10:", num_error_10_flipcy, num_10*mlc_error_rate["error_10"])
    return weight_torch

# 00, 01, 11, 10
def helmet(weight, weight_type, mlc_error_rate, name, tensors, num_bits, encode):
    shape = weight.shape
    MLC = weight_conf(weight, weight_type, num_bits)
    tensor_01, tensor_10, tensor_11 = tensors
    tensor_11 = tensor_11.cpu().numpy()
    tensor_01 = tensor_01.cpu().numpy()
    tensor_10 = tensor_10.cpu().numpy()
    if encode:
        encoded_weight = helmet_en(MLC.weight.to("cuda"), num_bits)
        if not os.path.isdir(f"./helmet_en/quantized-{args.model}-{num_bits}b"):
            os.mkdir(f"./helmet_en/quantized-{args.model}-{num_bits}b")
        torch.save(encoded_weight, f"./helmet_en/quantized-{args.model}-{num_bits}b/{name}.pt")
        weight_torch = encoded_weight.reshape(shape).to(weight.device)
    else:
        if num_bits == 8:
            dtype = np.uint8
        elif num_bits == 16:
            dtype = np.int16
        assert os.path.isdir(f"./helmet_en/quantized-{args.model}-{num_bits}b"), "You need to do encoding first"
        encoded_weight = torch.load(f"./helmet_en/quantized-{args.model}-{num_bits}b/{name}.pt")
        encoded_weight = encoded_weight.cpu().numpy().astype(dtype)
        weight = weight.view(-1).cpu().numpy().astype(dtype)

        num_11_helmet, _ = count_orig(encoded_weight, tensor_11, tensor_11, num_bits)
        num_error_11_helmet = int(mlc_error_rate["error_11"] * num_11_helmet)
        num_01_helmet, _ = count_orig(encoded_weight, tensor_01, tensor_11, num_bits)
        num_error_01_helmet = int(mlc_error_rate["error_10"] * num_01_helmet)

        # num_11, _  = count_orig(weight, tensor_11)
        # num_10, _ = count_orig(weight, tensor_10, tensor_11)
        num_error = (num_error_11_helmet, num_error_01_helmet)
        error_weight = inject_error(weight, num_error, mlc_error_rate, num_bits)
        weight_torch = error_weight.reshape(shape)
        # print("Number of error 11:", num_error_11_helmet, num_11*mlc_error_rate["error_11"])
        # print("Number of error 10:", num_error_10_helmet, num_10*mlc_error_rate["error_10"])
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

