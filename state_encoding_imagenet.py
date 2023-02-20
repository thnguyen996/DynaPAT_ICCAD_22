from collections import OrderedDict
from contextlib import suppress
from flipcy_quantized import flipcy_en, count, inject_error
from helmet_quantized import helmet_en
from matplotlib import font_manager
from matplotlib import rc
from matplotlib import rcParams
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from weight_quantized_conf import weight_conf
import argparse
import copy
import csv
import datetime
import flipcy_quantized as flipcy_quan
import glob
import logging
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import weight_quantized_conf_method3 as method3

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')
# _logger.setLevel(logging.INFO)
# handler = logging.FileHandler(f"./logs/{time}-{args.model}-imagenet-{args.method}")
# formatter = logging.Formatter('%(asctime)s : : %(message)s')
# handler.setFormatter(formatter)
# _logger.addHandler(handler)

torch.set_printoptions(profile="full")
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')

parser.add_argument("--gpu", default="0", type=str, help="GPU ids")
parser.add_argument("--num_bits", default=8, type=int, help="Number of quantized bits")
parser.add_argument("--method", default="proposed_method", type=str, help="Running method")
parser.add_argument("--save_data", "-s", action="store_true", help="Save the data")
parser.add_argument("--encode", "-e", action="store_true", help="Enable encode for flipcy and helmet")
parser.add_argument("--name", default="Name", type=str, help="Name of run")


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if not args.no_test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))


    dataset = create_dataset(
        root=args.data, name=args.dataset, split=args.split,
        load_bytes=args.tf_preprocessing, class_map=args.class_map)


    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    model.eval()
    if args.num_bits == 8:
        dtype = np.int8
    else:
        dtype = np.uint16
    error_11_pd = pd.read_csv("./error_level3_1year.csv", header=None, delimiter="\t")
    error_10_pd = pd.read_csv("./error_level2_1year.csv", header=None, delimiter="\t")
    iteration = 20
    tensorboard = args.save_data
    save_data = args.save_data

    if save_data:
        df = pd.DataFrame(columns=["Time", "Acc."])
    if tensorboard:
        ran = random.randint(1, 100)
        writer = SummaryWriter(f"./runs/{args.name}-{ran}")
        _logger.info(f"Run ID: {ran}")

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

    tensors = (tensor_00, tensor_01, tensor_10, tensor_11)
    with torch.no_grad():
        for (name, weight) in tqdm(model.named_parameters(), desc="Counting pattern: ", leave=False):
            if ( "weight" in name ) and ( "bn" not in name ) :
                # Dynamic fixed-point quantization
                sf = args.num_bits - 1. - compute_integral_part(weight, overflow_rate=0.0)
                quantized_weight, delta = linear_quantize(weight, sf, bits=args.num_bits)
                shape = weight.shape
                quantized_weight = quantized_weight.view(-1).detach().cpu().numpy()
                quantized_weight = quantized_weight.astype(dtype)

                num_00 = np.zeros_like(index_bit)
                num_01 = np.zeros_like(index_bit)
                num_10 = np.zeros_like(index_bit)
                num_11 = np.zeros_like(index_bit)

                _, index_11 = count(quantized_weight, tensor_11, tensor_11, index_bit, num_bits=args.num_bits)
                _, index_01 = count(quantized_weight, tensor_01, tensor_11, index_bit, num_bits=args.num_bits)
                _, index_10 = count(quantized_weight, tensor_10, tensor_11, index_bit, num_bits=args.num_bits)
                _, index_00 = count(quantized_weight, tensor_00, tensor_11, index_bit, num_bits=args.num_bits)

                num_11_count = np.unique(index_11[:, 1], return_counts=True)
                num_01_count = np.unique(index_01[:, 1], return_counts=True)
                num_10_count = np.unique(index_10[:, 1], return_counts=True)
                num_00_count = np.unique(index_00[:, 1], return_counts=True)

                num_11_count_index = (num_11_count[0]/2).astype(np.uint8)
                num_01_count_index = (num_01_count[0]/2).astype(np.uint8)
                num_10_count_index = (num_10_count[0]/2).astype(np.uint8)
                num_00_count_index = (num_00_count[0]/2).astype(np.uint8)

                num_11[num_11_count_index] = num_11_count[1]
                num_10[num_10_count_index] = num_10_count[1]
                num_01[num_01_count_index] = num_01_count[1]
                num_00[num_00_count_index] = num_00_count[1]

                total = num_11 + num_01 + num_10 + num_00
                num_11 = num_11/total*100
                num_10 = num_10/total*100
                num_01 = num_01/total*100
                num_00 = num_00/total*100
                total11.append(num_11)
                total10.append(num_10)
                total01.append(num_01)
                total00.append(num_00)

    total11 = np.stack(total11)
    total01 = np.stack(total01)
    total10 = np.stack(total10)
    total00 = np.stack(total00)
    total = np.stack((total00, total01, total10, total11))

    index_sort = np.argsort(total, 0)
    max_pattern_map = index_sort[3]
    state_encode = np.empty((max_pattern_map.shape[0], 4), dtype=np.uint8)

    for layer in range(max_pattern_map.shape[0]):
        msb1 = max_pattern_map[layer, 3]
        if msb1 == max_pattern_map[layer, 2]:
            msb2 = index_sort[2, layer, 2]
        else:
            msb2 = max_pattern_map[layer, 2]
        if max_pattern_map[layer, 1] == msb1 or max_pattern_map[layer, 1] == msb2:
            if index_sort[2, layer, 1] == msb1 or index_sort[2, layer, 1] == msb2:
                if index_sort[1, layer, 1] == msb1 or index_sort[1, layer, 1] == msb2:
                    lsb1 = index_sort[0, layer, 1]
                else:
                    lsb1 = index_sort[1, layer, 1]
            else:
                lsb1 = index_sort[2, layer, 1]
        else:
            lsb1 = max_pattern_map[layer, 1]
        for i in range(4):
            if i != msb1 and i != msb2 and i != lsb1:
                lsb2 = i
        state_encode[layer] = [msb1, msb2, lsb1, lsb2]

    np.save(f"./state_stats/{args.model}-imagenet-state-stats.npy", state_encode)
    print(f"Save state stats to ./state_stats/{args.model}-imagenet-state-stats.npy")

def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)
    clipped_value = torch.clamp(rounded, min_val, max_val)

    # clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value, delta

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    # if isinstance(v, Variable):
    #     v = v.data.cpu().numpy()[0]
    sf = math.ceil(math.log2(v+1e-12))
    return sf

def baseline(weight, mlc_error_rate, num_bits, tensors):
    if num_bits == 8:
        dtype = np.int8
    else:
        dtype = np.uint16
    MLC = weight_conf(weight, num_bits, method="baseline")
    error_weight = MLC.inject_error(mlc_error_rate)
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

def eval(args, data_config, model, loader, amp_autocast, crop_pct):
    param_count = sum([m.numel() for m in model.parameters()])
    # _logger.info('Model %s created, param count: %d' % (args.model, param_count))
    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
    input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.channels_last:
        input = input.contiguous(memory_format=torch.channels_last)
    model(input)
    end = time.time()
    loader_bar = tqdm(loader, leave=False)
    for batch_idx, (input, target) in enumerate(loader_bar):
        if args.no_prefetcher:
            target = target.cuda()
            input = input.cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # compute output
        with amp_autocast():
            output = model(input)

        if valid_labels is not None:
            output = output[:, valid_labels]
        loss = criterion(output, target)

        if real_labels is not None:
            real_labels.add_result(output)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loader_bar.set_description(f"Acc. : {top1.avg:.3f}")

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    # _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
    #    results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results['top1']

def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    time = datetime.datetime.now()
    setup_default_logging(log_path=f"./logs/{time}-{args.model}-imagenet-{args.method}")

    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
