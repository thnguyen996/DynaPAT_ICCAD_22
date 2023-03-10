"""Test quantized model"""
from cifar10_models import *
from flipcy_quantized import flipcy_en, count, inject_error
from functools import partial
from helmet_quantized import helmet_en
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import progress_bar
from weight_quantized_conf import weight_conf
import argparse
import copy
import flipcy_quantized as flipcy_quan
import logging
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.quantization
import torchvision
import torchvision.transforms as transforms
import traceback
import weight_quantized_conf_method3 as method3

torch.set_printoptions(profile="full")
np.set_printoptions(suppress=True)

msglogger = logging.getLogger()
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--mlc", default=8, type=int, help="Number of mlc bits")
parser.add_argument("--name", default="Name", type=str, help="Name of run")
parser.add_argument("--method", default="proposed_method", type=str, help="Running method")
parser.add_argument("--model", default="resnet18", type=str, help="Model")
parser.add_argument("--gpu", default="0", type=str, help="GPU ids")
parser.add_argument("--error_pat", default="00", type=str, help="error pattern")
parser.add_argument("--des_pat", default="00", type=str, help="destination pattern")
parser.add_argument("--save_data", "-s", action="store_true", help="Save the data")
parser.add_argument("--encode", "-e", action="store_true", help="Enable encode for flipcy and helmet")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root="~/Datasets/cifar10/", train=False, download=False, transform=transform_test
    )
    _, val_set = torch.utils.data.random_split(testset, [9500, 500])

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=2)

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
    elif args.model == "vgg16":
        net = vgg16_bn()
        net.load_state_dict(torch.load("./checkpoint/vgg16_bn.pt"))

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    orig_state_dict = copy.deepcopy(net.state_dict())
    error_11_pd = pd.read_csv("./error_level3_1year.csv", header=None, delimiter="\t")
    error_10_pd = pd.read_csv("./error_level2_1year.csv", header=None, delimiter="\t")
    # error_11_pd = error_11_pd.loc[0, 8:]
    # error_10_pd = error_10_pd.loc[0, 8:]

    iteration = 100
    weight_type = {"MLC": args.mlc, "SLC": args.num_bits - args.mlc}
    tensorboard = args.save_data
    save_data = args.save_data
    if save_data:
        df = pd.DataFrame(columns=["Time", "Acc."])
    if tensorboard:
        ran = random.randint(1, 100)
        writer = SummaryWriter(f"./runs/{args.name}-{ran}")
        print(f"Run ID: {ran}")

    list_01 = [1]
    list_10 = [2]
    list_11 = [3]
    list_00 = [0]
    for shift in range(2, args.num_bits, 2):
        next_pos = 2 ** (shift)
        list_01.append(next_pos)
        list_10.append(2 * next_pos)
        list_11.append(3 * next_pos)
        list_00.append(0)
    if args.num_bits == 8:
        dtype = np.int8
    else:
        dtype = np.uint16

    tensor_11 = np.array(list_11, dtype=dtype)
    tensor_10 = np.array(list_10, dtype=dtype)
    tensor_01 = np.array(list_01, dtype=dtype)
    tensor_00 = np.array(list_00, dtype=dtype)

    tensors = (tensor_00, tensor_01, tensor_10, tensor_11)
    if args.method == "proposed_method":
        state_encode = np.load(f"./state_stats/{args.model}-state-stats-fixed-point.npy")

    with torch.no_grad():
        count = 0
        for (name1, value1), (name2, value2) in zip(error_11_pd.items(), error_10_pd.items()):
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
                layer_index = 0
                # for name, weight in tqdm(net.named_parameters(), desc="Executing method:", leave=False):
                for name, weight in net.named_parameters():
                    if ("weight" in name) and ("bn" not in name):
                        # Fixed point quantization
                        if args.num_bits == 8:
                            qi, qf = (2, 6)
                        elif args.num_bits == 10:
                            qi, qf = (2, 8)
                        (imin, imax) = (-np.exp2(qi - 1), np.exp2(qi - 1) - 1)
                        fdiv = np.exp2(-qf)
                        quantized_weight = torch.round(torch.div(weight, fdiv))
                        shape = weight.shape

                        # Shift bit weights to 16-bit
                        quantized_weight = quantized_weight.view(-1).detach().cpu().numpy()
                        if args.num_bits == 10:
                            quantized_weight = (quantized_weight * np.exp2(6)).astype(dtype)
                            quantized_weight = (quantized_weight / np.exp2(6)).astype(dtype)
                        else:
                            quantized_weight = quantized_weight.astype(dtype)

                        mlc_error_rate = {"error_level3": error_11, "error_level2": error_10}
                        # mlc_error_rate = {"error_level3" : 0.01, "error_level2": 0.32}
                        if args.method == "proposed_method":
                            state_order = state_encode[layer_index, :]
                            error_quantized_weight = proposed_method(
                                quantized_weight, mlc_error_rate, args.num_bits, tensors, state_order
                            )
                        if args.method == "baseline":
                            error_quantized_weight = baseline(
                                quantized_weight, mlc_error_rate, args.num_bits, tensors
                            )
                        if args.method == "flipcy":
                            error_quantized_weight = flipcy(
                                quantized_weight,
                                mlc_error_rate,
                                name,
                                tensors,
                                num_bits=args.num_bits,
                                encode=args.encode,
                            )
                        if args.method == "helmet":
                            error_quantized_weight = helmet(
                                quantized_weight,
                                mlc_error_rate,
                                name,
                                tensors,
                                num_bits=args.num_bits,
                                encode=args.encode,
                            )
                        # if args.method == "test_case":
                        #     error_quantized_weight = test_case(
                        #         quantized_weight, mlc_error_rate, args.num_bits, tensors, args.error_pat, args.des_pat
                        #     )

                        error_quantized_weight  = torch.from_numpy(error_quantized_weight.astype(float))
                        if args.num_bits == 10:
                            error_quantized_weight[(error_quantized_weight > 511.0).nonzero()] = (
                                error_quantized_weight[(error_quantized_weight > 511.0).nonzero()] - 2**10
                            )
                        error_quantized_weight = error_quantized_weight.reshape(shape).cuda()
                        # Dequantization:
                        dequantized_weight = torch.clamp(torch.mul(error_quantized_weight, fdiv), min=imin, max=imax)
                        weight.copy_(dequantized_weight)
                        layer_index += 1

                class_acc = test(net, criterion, optimizer, testloader, device)
                running_acc.append(class_acc)
            avr_acc = sum(running_acc) / len(running_acc)
            if tensorboard and not args.encode:
                writer.add_scalar("Acc./", avr_acc, count)
                writer.close()
            if save_data and not args.encode:
                df.loc[count] = [count, avr_acc]
                df.to_csv(f"./results-2022/{args.name}.csv", mode="w", header=True)
            count += 1

def test_case(weight, mlc_error_rate, num_bits, tensors, error_pat, des_pat):
    if num_bits == 8:
        dtype = np.int8
    else:
        dtype = np.uint16
    orig_weight = np.copy(weight)
    error_weight = method3.inject_error(weight, orig_weight, mlc_error_rate["error_level3"], error_pat, des_pat, num_bits)

    return error_weight

def proposed_method(weight, mlc_error_rate, num_bits, tensors, state_order):
    if num_bits == 8:
        dtype = np.int8
    else:
        dtype = np.uint16

    state = ("00", "01", "10", "11")
    level2 = state[state_order[2]]
    level3 = state[state_order[3]]
    level4 = state[state_order[0]]
    orig_weight = np.copy(weight)
    error_weight = method3.inject_error(weight, orig_weight, mlc_error_rate["error_level3"], level3, level4, num_bits)
    error_weight = method3.inject_error(error_weight, orig_weight, mlc_error_rate["error_level2"], level2, level3, num_bits)

    return error_weight


def baseline(weight, mlc_error_rate, num_bits, tensors):
    if num_bits == 8:
        dtype = np.int8
    else:
        dtype = np.uint16
    MLC = weight_conf(weight, num_bits, method="baseline")
    error_weight = MLC.inject_error(mlc_error_rate)
    return error_weight


# 00, 01, 11, 10
def flipcy(weight, mlc_error_rate, name, tensors, num_bits, encode):
    shape = weight.shape
    MLC = weight_conf(weight, num_bits, method="baseline")
    tensor_00, tensor_01, tensor_10, tensor_11 = tensors
    if encode:
        encoded_weight = flipcy_en(MLC.weight, num_bits, tensors)
        if not os.path.isdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b"):
            os.mkdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b")
        encoded_weight = torch.from_numpy(encoded_weight.astype(np.float32)).cuda()
        torch.save(encoded_weight.reshape(shape), f"./flipcy_en/quantized-{args.model}-{num_bits}b/{name}.pt")
        return weight
    else:
        if num_bits == 8:
            dtype = np.int8
        else:
            dtype = np.uint16
        flipcy_error_rate = {}
        assert os.path.isdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b"), "You need to do encoding first"
        encoded_weight = torch.load(f"./flipcy_en/quantized-{args.model}-{num_bits}b/{name}.pt")

        encoded_weight = encoded_weight.cpu().numpy().astype(dtype)
        encoded_weight = encoded_weight.reshape(int(encoded_weight.size / 4), -1)

        orig_weight = np.copy(MLC.weight.reshape(int(encoded_weight.size / 4), -1))

        num_01_flipcy = flipcy_quan.count(encoded_weight, tensor_01, tensor_11, num_bits)
        num_11_flipcy = flipcy_quan.count(encoded_weight, tensor_11, tensor_11, num_bits)

        num_01 = flipcy_quan.count(orig_weight, tensor_01, tensor_11, num_bits)
        num_11 = flipcy_quan.count(orig_weight, tensor_11, tensor_11, num_bits)

        num_01_flipcy = np.sum(np.sum(num_01_flipcy, axis=1), 0)
        num_11_flipcy = np.sum(np.sum(num_11_flipcy, axis=1), 0)

        num_01 = np.sum(np.sum(num_01, axis=1), 0)
        num_11 = np.sum(np.sum(num_11, axis=1), 0)

        num_error_01 = mlc_error_rate["error_level2"] * num_01_flipcy
        num_error_11 = mlc_error_rate["error_level3"] * num_11_flipcy

        if num_11 != 0:
            flipcy_error_rate["error_level3"] = num_error_11 / num_11
        else:
            flipcy_error_rate["error_level3"] = None
        if num_01 != 0:
            flipcy_error_rate["error_level2"] = num_error_01 / num_01
        else:
            flipcy_error_rate["error_level2"] = None

        error_weight = MLC.inject_error(flipcy_error_rate)
        # print("Flipcy error rate", flipcy_error_rate["error_level3"], flipcy_error_rate["error_level3"])
        return error_weight


# 00, 01, 11, 10
def helmet(weight, mlc_error_rate, name, tensors, num_bits, encode):
    shape = weight.shape
    MLC = weight_conf(weight, num_bits, method="baseline")
    tensor_00, tensor_01, tensor_10, tensor_11 = tensors
    helmet_error_rate = {}
    if encode:
        encoded_weight = helmet_en(MLC.weight, num_bits)
        if not os.path.isdir(f"./helmet_en/quantized-{args.model}-{num_bits}b"):
            os.mkdir(f"./helmet_en/quantized-{args.model}-{num_bits}b")
        encoded_weight = torch.from_numpy(encoded_weight.astype(np.float32)).cuda()
        torch.save(encoded_weight, f"./helmet_en/quantized-{args.model}-{num_bits}b/{name}.pt")
        return weight
    else:
        if num_bits == 8:
            dtype = np.int8
        else:
            dtype = np.uint16
        assert os.path.isdir(f"./helmet_en/quantized-{args.model}-{num_bits}b"), "You need to do encoding first"
        encoded_weight = torch.load(f"./helmet_en/quantized-{args.model}-{num_bits}b/{name}.pt")

        encoded_weight = encoded_weight.cpu().numpy().astype(dtype)
        encoded_weight = encoded_weight.reshape(int(encoded_weight.size / 4), -1)

        orig_weight = np.copy(MLC.weight.reshape(int(encoded_weight.size / 4), -1))

        num_01_helmet = flipcy_quan.count(encoded_weight, tensor_01, tensor_11, num_bits)
        num_11_helmet = flipcy_quan.count(encoded_weight, tensor_11, tensor_11, num_bits)

        num_01 = flipcy_quan.count(orig_weight, tensor_01, tensor_11, num_bits)
        num_11 = flipcy_quan.count(orig_weight, tensor_11, tensor_11, num_bits)

        num_01_helmet = np.sum(np.sum(num_01_helmet, axis=1), 0)
        num_11_helmet = np.sum(np.sum(num_11_helmet, axis=1), 0)

        num_01 = np.sum(np.sum(num_01, axis=1), 0)
        num_11 = np.sum(np.sum(num_11, axis=1), 0)

        num_error_01 = mlc_error_rate["error_level2"] * num_01_helmet
        num_error_11 = mlc_error_rate["error_level3"] * num_11_helmet
        if num_11 != 0:
            helmet_error_rate["error_level3"] = num_error_11 / num_11
        else:
            helmet_error_rate["error_level3"] = None
        if num_01 != 0:
            helmet_error_rate["error_level2"] = num_error_01 / num_01
        else:
            helmet_error_rate["error_level2"] = None

        error_weight = MLC.inject_error(helmet_error_rate)
        error_weight = error_weight.reshape(weight.shape)

        # print(f"num 11 helmet: {num_11_helmet}, num 11 orig: {num_11}")
        # print(f"num 01 helmet: {num_01_helmet}, num 01 orig: {num_01}")
        return error_weight


def count_00(weight, tensor_00, tensor_11, num_bits):
    index_bit = np.arange(0, num_bits, 2)
    num_00 = 0
    indices_00 = []
    for tensor_00_i, tensor_11_i, index_b in zip(tensor_00, tensor_11, index_bit):
        and_result = np.bitwise_and(tensor_11_i, weight)
        index_00 = (and_result == tensor_00_i).nonzero()[0]
        bit_index = np.full_like(index_00, index_b)
        bit_index = np.transpose(np.expand_dims(bit_index, 0), (1, 0))
        index_00 = np.transpose(np.expand_dims(index_00, 0), (1, 0))
        index_tensor = np.concatenate((index_00, bit_index), axis=1)
        indices_00.append(index_tensor)
        num_00 += index_00.shape[1]
    total_index_00 = np.concatenate(indices_00, axis=0)
    indices = np.unique(total_index_00[:, 0], return_counts=True)
    zeros = np.zeros(weight.size)
    zeros[indices[0]] = indices[1]
    return zeros


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

def circshift(weight, num_bits):
    if num_bits == 16:
        weight_np = weight.view(np.uint16)
        save_bit = np.left_shift(weight_np, 15)
        rot_bit = np.right_shift(weight_np, 1)
        rot_weight = np.bitwise_or(save_bit, rot_bit).view(np.int16)
    elif num_bits == 8:
        weight_np = weight
        save_bit = np.left_shift(weight_np, 7)
        rot_bit = np.right_shift(weight_np, 1)
        rot_weight = np.bitwise_or(save_bit, rot_bit)
    return rot_weight

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
