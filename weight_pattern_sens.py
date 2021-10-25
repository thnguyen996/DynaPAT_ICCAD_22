import torch
import cupy as cp
import numpy as np

torch.set_printoptions(profile="full")

class weight_conf(object):
    def __init__(self, weight):
        self.shape = weight.shape
        self.device = weight.device
        self.weight = weight.view(-1).cpu()

    def inject_error(self, mlc_error_rate, error_pat, des_pat):
        weight = cp.asarray(self.weight)
        weight = weight.view(cp.uint32)

        # For debugging:
        # error_pat = "10"
        # des_pat = "11"

        list_00 = [0]
        list_01 = [1]
        list_10 = [2]
        list_11 = [3]
        for shift in range(2, 32, 2):
            next_pos = 2 ** (shift)
            list_00.append(0)
            list_01.append(next_pos)
            list_10.append(2 * next_pos)
            list_11.append(3 * next_pos)
        tensor_11 = cp.asarray(list_11, dtype=cp.uint32)
        tensor_10 = cp.asarray(list_10, dtype=cp.uint32)
        tensor_01 = cp.asarray(list_01, dtype=cp.uint32)
        tensor_00 = cp.asarray(list_00, dtype=cp.uint32)
        tensor_00_inv = cp.invert(tensor_11)
        tensor_01_inv = cp.invert(tensor_10)
        tensor_10_inv = cp.invert(tensor_01)
        index_bit = cp.arange(0, 32, 2)

        if error_pat == "00":
            num_00, index_00 = count_pattern(weight, tensor_00, tensor_11, index_bit)
            error_rate = mlc_error_rate["error_rate"]
            num_error_00 = int(error_rate*num_00)
            error00_randn_index = cp.random.permutation(num_00)[:num_error_00]
            error00_indices = index_00[error00_randn_index, :]
            tensor10_index = tensor_10[(error00_indices[:, 1] / 2).astype(cp.uint32)]
            tensor01_index = tensor_01[(error00_indices[:, 1] / 2).astype(cp.uint32)]
            tensor11_index = tensor_11[(error00_indices[:, 1] / 2).astype(cp.uint32)]
            weight_np = cp.asnumpy(weight)
            error00_indices_np = cp.asnumpy(error00_indices)
            if des_pat == "01":
                tensor01_index_np = cp.asnumpy(tensor01_index)
                np.bitwise_or.at(weight_np, error00_indices_np[:, 0], tensor01_index_np)
                weight = cp.asarray(weight_np)
            elif des_pat == "10":
                tensor10_index_np = cp.asnumpy(tensor10_index)
                np.bitwise_or.at(weight_np, error00_indices_np[:, 0], tensor10_index_np)
                weight = cp.asarray(weight_np)
            elif des_pat == "11":
                tensor11_index_np = cp.asnumpy(tensor11_index)
                np.bitwise_or.at(weight_np, error00_indices_np[:, 0], tensor11_index_np)
                weight = cp.asarray(weight_np)

        if error_pat == "01":
            num_01, index_01 = count_pattern(weight, tensor_01, tensor_11, index_bit)
            error_rate = mlc_error_rate["error_rate"]
            num_error_01 = int(error_rate*num_01)
            error01_randn_index = cp.random.permutation(num_01)[:num_error_01]
            error01_indices = index_01[error01_randn_index, :]
            tensor10_index = tensor_10[(error01_indices[:, 1] / 2).astype(cp.uint32)]
            tensor00_inv_index = tensor_00_inv[(error01_indices[:, 1] / 2).astype(cp.uint32)]
            tensor11_index = tensor_11[(error01_indices[:, 1] / 2).astype(cp.uint32)]
            weight_np = cp.asnumpy(weight)
            error01_indices_np = cp.asnumpy(error01_indices)
            if des_pat == "00":
                tensor00_inv_index_np = cp.asnumpy(tensor00_inv_index)
                np.bitwise_and.at(weight_np, error01_indices_np[:, 0], tensor00_inv_index_np)
                weight = cp.asarray(weight_np)
            elif des_pat == "10":
                tensor11_index_np = cp.asnumpy(tensor11_index)
                np.bitwise_xor.at(weight_np, error01_indices_np[:, 0], tensor11_index_np)
                weight = cp.asarray(weight_np)
            elif des_pat == "11":
                tensor11_index_np = cp.asnumpy(tensor11_index)
                np.bitwise_or.at(weight_np, error01_indices_np[:, 0], tensor11_index_np)
                weight = cp.asarray(weight_np)

        if error_pat == "10":
            num_10, index_10 = count_pattern(weight, tensor_10, tensor_11, index_bit)
            error_rate = mlc_error_rate["error_rate"]
            num_error_10 = int(error_rate*num_10)
            error10_randn_index = cp.random.permutation(num_10)[:num_error_10]
            error10_indices = index_10[error10_randn_index, :]
            tensor00_index = tensor_00[(error10_indices[:, 1] / 2).astype(cp.uint32)]
            tensor01_index = tensor_01[(error10_indices[:, 1] / 2).astype(cp.uint32)]
            tensor11_index = tensor_11[(error10_indices[:, 1] / 2).astype(cp.uint32)]
            tensor00_inv_index = tensor_00_inv[(error10_indices[:, 1] / 2).astype(cp.uint32)]
            weight_np = cp.asnumpy(weight)
            error10_indices_np = cp.asnumpy(error10_indices)
            if des_pat == "01":
                tensor11_index_np = cp.asnumpy(tensor11_index)
                np.bitwise_xor.at(weight_np, error10_indices_np[:, 0], tensor11_index_np)
                weight = cp.asarray(weight_np)
            elif des_pat == "00":
                tensor00_inv_index_np = cp.asnumpy(tensor00_inv_index)
                np.bitwise_and.at(weight_np, error10_indices_np[:, 0], tensor00_inv_index_np)
                weight = cp.asarray(weight_np)
            elif des_pat == "11":
                tensor11_index_np = cp.asnumpy(tensor11_index)
                np.bitwise_or.at(weight_np, error10_indices_np[:, 0], tensor11_index_np)
                weight = cp.asarray(weight_np)

        if error_pat == "11":
            num_11, index_11 = count_pattern(weight, tensor_11, tensor_11, index_bit)
            error_rate = mlc_error_rate["error_rate"]
            num_error_11 = int(error_rate*num_11)
            error11_randn_index = cp.random.permutation(num_11)[:num_error_11]
            error11_indices = index_11[error11_randn_index, :]
            tensor10_index = tensor_10[(error11_indices[:, 1] / 2).astype(cp.uint32)]
            tensor01_index = tensor_01[(error11_indices[:, 1] / 2).astype(cp.uint32)]
            tensor00_index = tensor_00[(error11_indices[:, 1] / 2).astype(cp.uint32)]
            tensor01_inv_index = tensor_01_inv[(error11_indices[:, 1] / 2).astype(cp.uint32)]
            tensor10_inv_index = tensor_10_inv[(error11_indices[:, 1] / 2).astype(cp.uint32)]
            tensor00_inv_index = tensor_00_inv[(error11_indices[:, 1] / 2).astype(cp.uint32)]
            weight_np = cp.asnumpy(weight)
            error11_indices_np = cp.asnumpy(error11_indices)
            if des_pat == "01":
                tensor01_inv_index_np = cp.asnumpy(tensor01_inv_index)
                np.bitwise_and.at(weight_np, error11_indices_np[:, 0], tensor01_inv_index_np)
                weight = cp.asarray(weight_np)
            elif des_pat == "10":
                tensor10_inv_index_np = cp.asnumpy(tensor10_inv_index)
                np.bitwise_and.at(weight_np, error11_indices_np[:, 0], tensor10_inv_index_np)
                weight = cp.asarray(weight_np)
            elif des_pat == "00":
                tensor00_inv_index_np = cp.asnumpy(tensor00_inv_index)
                np.bitwise_and.at(weight_np, error11_indices_np[:, 0], tensor00_inv_index_np)
                weight = cp.asarray(weight_np)

        # Convert to 32 bit floating point pytorch tensor
        weight_float = weight.view(cp.float32)
        weight_error = torch.tensor(weight_float)
        # weight_float[(weight_float < 0.).nonzero()] = weight_float[(weight_float < 0.).nonzero()] + 2**16

        return weight_error.to(self.device)

def count_pattern(weight, tensor, tensor_11, index_bit):
    indices = []
    num = 0
    for tensor_i, tensor_11_i, index_b in zip(tensor, tensor_11, index_bit):
        and_result = cp.bitwise_and(tensor_11_i, weight)
        index = (and_result == tensor_i).nonzero()[0]
        bit_index = cp.full(index.size, index_b)
        index_tensor = cp.stack((index, bit_index))
        indices.append(index_tensor)
        num += index.size
    total_index = cp.concatenate(indices, axis=1)
    total_index = total_index.transpose(1, 0)
    return num, total_index
