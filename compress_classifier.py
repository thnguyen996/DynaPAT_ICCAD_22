#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagemodel).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    compression_scheduler.on_epoch_end(epoch)
    save_checkpoint()

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilemodel
"""

from distiller.data_loggers import *
from distiller.models import create_model
from distiller.utils import float_range_argparse_checker as float_range
from flipcy_quantized import flipcy_en, count_orig, inject_error
from functools import partial
from helmet_quantized import helmet_en
from ptq_lapq import image_classifier_ptq_lapq
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from weight_quantized_conf import *
import copy
import distiller
import distiller.apputils as apputils
import distiller.apputils.image_classifier as classifier
import distiller.quantization as quantization
import logging
import numpy as np
import os
import pandas as pd
import parser
import random
import traceback
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Logger handle
msglogger = logging.getLogger()


def main():
    # Parse arguments
    args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser(True)).parse_args()
    app = ClassifierCompressorSampleApp(args, script_dir=os.path.dirname(__file__))
    if app.handle_subapps():
        return
    app.run_training_loop()
    # Finally run results on the test set
    return app.test()

def handle_subapps(model, criterion, optimizer, compression_scheduler, pylogger, args):
    def load_test_data(args):
        test_loader = classifier.load_data(args, load_train=False, load_val=False, load_test=True)
        return test_loader
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
            if not os.path.isdir(f"./flipcy_en/quantized-{args.arch}-{num_bits}b-imagenet"):
                os.mkdir(f"./flipcy_en/quantized-{args.arch}-{num_bits}b-imagenet")
            torch.save(encoded_weight, f"./flipcy_en/quantized-{args.arch}-{num_bits}b-imagenet/{name}.pt")
            weight_torch = encoded_weight.reshape(shape)
        else:
            if num_bits == 8:
                dtype = np.uint8
            elif num_bits == 16:
                dtype = np.int16
            assert os.path.isdir(f"./flipcy_en/quantized-{args.arch}-{num_bits}b-imagenet"), "You need to do encoding first"
            encoded_weight = torch.load(f"./flipcy_en/quantized-{args.arch}-{num_bits}b-imagenet/{name}.pt")

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
            if not os.path.isdir(f"./helmet_en/quantized-{args.arch}-{num_bits}b-imagenet"):
                os.mkdir(f"./helmet_en/quantized-{args.arch}-{num_bits}b-imagenet")
            torch.save(encoded_weight, f"./helmet_en/quantized-{args.arch}-{num_bits}b-imagenet/{name}.pt")
            weight_torch = encoded_weight.reshape(shape).to(weight.device)
        else:
            if num_bits == 8:
                dtype = np.uint8
            elif num_bits == 16:
                dtype = np.int16
            assert os.path.isdir(f"./helmet_en/quantized-{args.arch}-{num_bits}b-imagenet"), "You need to do encoding first"
            encoded_weight = torch.load(f"./helmet_en/quantized-{args.arch}-{num_bits}b-imagenet/{name}.pt")
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

    do_exit = False
    if args.qe_calibration and not (args.evaluate and args.quantize_eval):
        classifier.acts_quant_stats_collection(model, criterion, pylogger, args, save_to_file=True)
        do_exit = True
    elif args.evaluate:
        if args.quantize_eval and args.qe_lapq:
            image_classifier_ptq_lapq(model, criterion, pylogger, args)
        else:
            test_loader = load_test_data(args)
            classifier.evaluate_model(test_loader, model, criterion, pylogger,
                classifier.create_activation_stats_collectors(model, *args.activation_stats),
                args, scheduler=compression_scheduler)
        do_exit = True
    elif args.run:
        test_loader = load_test_data(args)
        if hasattr(model, 'quantizer_metadata') and \
                model.quantizer_metadata['type'] == distiller.quantization.PostTrainLinearQuantizer:
            raise RuntimeError('Trying to invoke post-training quantization on a model that has already been post-'
                               'train quantized. Model was likely loaded from a checkpoint. Please run again without '
                               'passing the --quantize-eval flag')
        if not (args.qe_dynamic or args.qe_stats_file or args.qe_config_file):
            args_copy = copy.deepcopy(args)
            args_copy.qe_calibration = args.qe_calibration if args.qe_calibration is not None else 0.05

            # set stats into args stats field
            args.qe_stats_file = acts_quant_stats_collection(
                model, criterion, loggers, args_copy, save_to_file=save_flag)

        args_qe = copy.deepcopy(args)

        quantizer = quantization.PostTrainLinearQuantizer.from_args(model, args_qe)
        dummy_input = distiller.get_dummy_input(input_shape=model.input_shape)
        quantizer.prepare_model(dummy_input)

        if args.qe_convert_pytorch:
            model = _convert_ptq_to_pytorch(model, args_qe)

        orig_state_dict = copy.deepcopy(model.state_dict())
        error_11_pd = pd.read_csv("./error_level3_1year.csv", header = None, delimiter="\t")
        error_10_pd = pd.read_csv("./error_level2_1year.csv", header = None, delimiter="\t")
        # error_11_pd = error_11_pd.loc[0, 8:]
        # error_10_pd = error_10_pd.loc[0, 8:]

        iteration = 20
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
                    model.load_state_dict(orig_state_dict)
                    if args.encode & (it > 0):
                        break
                    for name, weight in tqdm(model.named_parameters(), desc="Executing method:", leave=False):
                        if ( "weight" in name ) and ( "bn" not in name ):
                            # level = "bit"
                            mlc_error_rate = {"error_level3" : error_11, "error_level2": error_10}
                            if args.method == "proposed_method":
                                error_weight = proposed_method(weight, weight_type, mlc_error_rate, args.num_bits)
                            if args.method == "flipcy":
                                error_weight = flipcy(weight, weight_type, mlc_error_rate, name, tensors, num_bits=args.num_bits, encode=args.encode)
                            if args.method == "helmet":
                                error_weight = helmet(weight, weight_type, mlc_error_rate, name, tensors, num_bits=args.num_bits, encode=args.encode)
                            weight.copy_(error_weight)

                    class_acc = classifier.test(test_loader, model, criterion, pylogger, args=args_qe)[0]
                    running_acc.append(class_acc)
                avr_acc = sum(running_acc) / len(running_acc)
                if tensorboard and not args.encode:
                    writer.add_scalar("Acc./", avr_acc, count)
                    writer.close()
                if save_data and not args.encode:
                    df.loc[count] = [count, avr_acc]
                    df.to_csv(f"./imagenet_results/{args.name}.csv", mode='w', header=True)
                count += 1

        # test_res = classifier.test(test_loader, model, criterion, pylogger, args=args_qe)[0]
        # print(f"Accuracy {test_res}")

        do_exit = True
    return do_exit


def early_exit_init(args):
    if not args.earlyexit_thresholds:
        return
    args.num_exits = len(args.earlyexit_thresholds) + 1
    args.loss_exits = [0] * args.num_exits
    args.losses_exits = []
    args.exiterrors = []
    msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)


class ClassifierCompressorSampleApp(classifier.ClassifierCompressor):
    def __init__(self, args, script_dir):
        super().__init__(args, script_dir)
        early_exit_init(self.args)
        # Save the randomly-initialized model before training (useful for lottery-ticket method)
        if args.save_untrained_model:
            ckpt_name = '_'.join((self.args.name or "", "untrained"))
            apputils.save_checkpoint(0, self.args.arch, self.model,
                                     name=ckpt_name, dir=msglogger.logdir)


    def handle_subapps(self):
        return handle_subapps(self.model, self.criterion, self.optimizer,
                              self.compression_scheduler, self.pylogger, self.args)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))

