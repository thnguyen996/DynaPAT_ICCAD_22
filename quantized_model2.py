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

torch.set_printoptions(profile="full")
msglogger = logging.getLogger()
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--mlc", default=8, type=int, help="Number of mlc bits")
parser.add_argument("--name", default="Name", type=str, help="Name of run")
parser.add_argument("--method", default="proposed_method", type=str, help="Running method")
parser.add_argument("--model", default="resnet18", type=str, help="Model")
parser.add_argument("--gpu", default="0", type=str, help="GPU ids")
parser.add_argument("--case", default="1", type=str, help="case run")
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
    list_00 = [0]
    for shift in range(2, args.num_bits, 2):
        next_pos = 2 ** (shift)
        list_01.append(next_pos)
        list_10.append(2 * next_pos)
        list_11.append(3 * next_pos)
        list_00.append(0)
    if args.num_bits == 16:
        dtype=np.uint16
    elif args.num_bits == 8:
        dtype=np.uint8
    tensor_11 = np.array(list_11, dtype=dtype)
    tensor_10 = np.array(list_10, dtype=dtype)
    tensor_01 = np.array(list_01, dtype=dtype)
    tensor_00 = np.array(list_00, dtype=dtype)

    tensors = (tensor_00, tensor_01, tensor_10, tensor_11)

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
                        mlc_error_rate = {"error_level3" : error_11, "error_level2": error_10}
                        if args.method == "proposed_method":
                            error_weight = proposed_method(weight, weight_type, mlc_error_rate, args.num_bits, tensors, args.case)
                        if args.method == "baseline":
                            error_weight = baseline(weight, weight_type, mlc_error_rate, args.num_bits, tensors, args.case)
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
                df.to_csv(f"./result/{args.name}.csv", mode='w', header=True)
            count += 1


def proposed_method(weight, weight_type, mlc_error_rate, num_bits, tensors, case):
    if num_bits == 8:
        dtype = np.uint8
    elif num_bits == 16:
        dtype = np.int16

    MLC = weight_conf(weight, weight_type, num_bits, method="proposed")
    error_weight = MLC.inject_error(mlc_error_rate)
    error_weight = error_weight.reshape(weight.shape)
    return error_weight

def baseline(weight, weight_type, mlc_error_rate, num_bits, tensors, case):
    if num_bits == 8:
        dtype = np.uint8
    elif num_bits == 16:
        dtype = np.int16

    MLC = weight_conf(weight, weight_type, num_bits, method="baseline")
    error_weight = MLC.inject_error(mlc_error_rate)
    error_weight = error_weight.reshape(weight.shape)
    return error_weight

def proposed_method_en(weight, num_bits, tensors):
    if num_bits == 8:
        dtype = np.uint8
    elif num_bits == 16:
        dtype = np.int16
    tensor_00, tensor_01, tensor_10, tensor_11 = tensors

    weight = weight.reshape(int(weight.numel() / 1), 1)
    orig_weight = weight.clone()
    weight = weight.cpu().numpy().astype(dtype)
    orig_weight = orig_weight.cpu().numpy().astype(dtype)

    # Flip 00 -> 01 and 11 -> 10:
    flipped_weight = flip_proposed(weight, orig_weight, tensors, num_bits)

    num_00_orig = count_00(weight, tensor_00, tensor_11, num_bits)
    num_11_orig = count_00(weight, tensor_11, tensor_11, num_bits)
    num_01_orig = count_00(weight, tensor_01, tensor_11, num_bits)
    num_10_orig = count_00(weight, tensor_10, tensor_11, num_bits)

    sum_0011 = num_00_orig + num_11_orig
    sum_0110 = num_01_orig + num_10_orig

    if num_bits == 16:
        weight = weight.astype(np.uint16)

    # if np.sum(sum_0011) > np.sum(sum_0110):
    #     return flipped_weight
    # else:
    #     return weight

    total= np.stack((sum_0011, sum_0110))
    min_case = np.argmin(total, axis=0)
    weight[(min_case == 1).nonzero()[0], :] = flipped_weight[(min_case == 1).nonzero()[0], :]
    return weight

    # total_00 = np.stack((num_00_orig, num_00_inv))
    # min_case = np.argmin(total_00, axis=0)
    # # weight = weight.reshape(int(weight.size / 32), -1)
    # weight[(min_case == 1).nonzero()[0], :] = inv_weight[(min_case == 1).nonzero()[0], :]
    # # weight = weight.reshape(int(weight.size / 1), 1)

    # weight_torch = torch.tensor(weight.astype(np.float32), device="cuda")

def flip_proposed(weight, orig_weight, tensors, num_bits):
    shape = weight.shape
    weight = weight.flatten()
    orig_weight = orig_weight.flatten()
    index_bit = np.arange(0, num_bits, 2)
    tensor_10_inv = np.invert(tensors[1])
    tensor_00_inv = np.invert(tensors[3])

    # Flip 00 --> 01
    num_00, index_00 = count(orig_weight, tensors[0], tensors[3], index_bit, num_bits)
    tensor01_index = tensors[1][(index_00[:, 1] / 2).astype(np.uint8)]
    np.bitwise_or.at(weight, index_00[:, 0], tensor01_index)
    # Flip 11 --> 10
    num_11, index_11 = count(orig_weight, tensors[3], tensors[3], index_bit, num_bits)
    tensor10_index = tensor_10_inv[(index_11[:, 1] / 2).astype(np.uint8)]
    np.bitwise_and.at(weight, index_11[:, 0], tensor10_index)
    # Flip 01 --> 00
    num_01, index_01 = count(orig_weight, tensors[1], tensors[3], index_bit, num_bits)
    tensor00_index = tensor_00_inv[(index_01[:, 1] / 2).astype(np.uint8)]
    np.bitwise_and.at(weight, index_01[:, 0], tensor00_index)
    # Flip 10 --> 11
    num_10, index_10 = count(orig_weight, tensors[2], tensors[3], index_bit, num_bits)
    tensor11_index = tensors[3][(index_10[:, 1] / 2).astype(np.uint8)]
    np.bitwise_or.at(weight, index_10[:, 0], tensor11_index)

    return weight.reshape(shape)


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

# 00, 01, 11, 10
def flipcy(weight, weight_type, mlc_error_rate, name, tensors, num_bits, encode):
    shape = weight.shape
    MLC = weight_conf(weight, weight_type, num_bits, method="baseline")
    tensor_00, tensor_01, tensor_10, tensor_11 = tensors
    if encode:
        encoded_weight = flipcy_en(MLC.weight, num_bits, tensors)
        if not os.path.isdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b"):
            os.mkdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b")
        torch.save(encoded_weight, f"./flipcy_en/quantized-{args.model}-{num_bits}b/{name}.pt")
        error_weight = encoded_weight.reshape(shape)
    else:
        if num_bits == 8:
            dtype = np.uint8
        elif num_bits == 16:
            dtype = np.int16
        assert os.path.isdir(f"./flipcy_en/quantized-{args.model}-{num_bits}b"), "You need to do encoding first"
        encoded_weight = torch.load(f"./flipcy_en/quantized-{args.model}-{num_bits}b/{name}.pt")

        encoded_weight = encoded_weight.cpu().numpy().astype(dtype)
        num_01_flipcy, _ = count_orig(encoded_weight, tensor_01, tensor_11, num_bits)
        num_10_flipcy, _ = count_orig(encoded_weight, tensor_10, tensor_11, num_bits)

        num_01, _ = count_orig(MLC.weight.cpu().numpy().astype(dtype), tensor_01, tensor_11, num_bits)
        num_10, _ = count_orig(MLC.weight.cpu().numpy().astype(dtype), tensor_10, tensor_11, num_bits)

        num_error_01 = mlc_error_rate["error_level3"] * num_01_flipcy
        num_error_10 = mlc_error_rate["error_level2"] * num_10_flipcy
        mlc_error_rate["error_level3"] = num_error_01/num_01
        mlc_error_rate["error_level2"] = num_error_10/num_10

        error_weight = MLC.inject_error(mlc_error_rate)
        error_weight = error_weight.reshape(weight.shape)
        # print("Number of error 11:", num_error_01, num_11*mlc_error_rate["error_11"])
        # print("Number of error 10:", num_error_10, num_10*mlc_error_rate["error_10"])
    return error_weight

# 00, 01, 11, 10
def helmet(weight, weight_type, mlc_error_rate, name, tensors, num_bits, encode):
    shape = weight.shape
    MLC = weight_conf(weight, weight_type, num_bits, method="baseline")
    tensor_00, tensor_01, tensor_10, tensor_11 = tensors
    if encode:
        encoded_weight = helmet_en(MLC.weight.to("cuda"), num_bits)
        if not os.path.isdir(f"./helmet_en/quantized-{args.model}-{num_bits}b"):
            os.mkdir(f"./helmet_en/quantized-{args.model}-{num_bits}b")
        torch.save(encoded_weight, f"./helmet_en/quantized-{args.model}-{num_bits}b/{name}.pt")
        error_weight = encoded_weight.reshape(shape).to(weight.device)
    else:
        if num_bits == 8:
            dtype = np.uint8
        elif num_bits == 16:
            dtype = np.int16
        assert os.path.isdir(f"./helmet_en/quantized-{args.model}-{num_bits}b"), "You need to do encoding first"
        encoded_weight = torch.load(f"./helmet_en/quantized-{args.model}-{num_bits}b/{name}.pt")

        encoded_weight = encoded_weight.cpu().numpy().astype(dtype)
        num_01_helmet, _ = count_orig(encoded_weight, tensor_01, tensor_11, num_bits)
        num_10_helmet, _ = count_orig(encoded_weight, tensor_10, tensor_11, num_bits)

        num_01, _ = count_orig(MLC.weight.cpu().numpy().astype(dtype), tensor_01, tensor_11, num_bits)
        num_10, _ = count_orig(MLC.weight.cpu().numpy().astype(dtype), tensor_10, tensor_11, num_bits)

        num_error_01 = mlc_error_rate["error_level3"] * num_01_helmet
        num_error_10 = mlc_error_rate["error_level2"] * num_10_helmet
        mlc_error_rate["error_level3"] = num_error_01/num_01
        mlc_error_rate["error_level2"] = num_error_10/num_10

        error_weight = MLC.inject_error(mlc_error_rate)
        error_weight = error_weight.reshape(weight.shape)
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

