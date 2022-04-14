import torch
import numpy as np

torch.set_printoptions(profile="full")

def inject_error(weight, orig_weight, error_rate, error_pat, des_pat, num_bits):
    if num_bits == 16:
        dtype = np.uint16
    elif num_bits == 8:
        dtype = np.uint8
    weight = weight.astype(dtype)
    orig_weight = orig_weight.astype(dtype)

    list_00 = [0]
    list_01 = [1]
    list_10 = [2]
    list_11 = [3]
    for shift in range(2, num_bits, 2):
        next_pos = 2 ** (shift)
        list_00.append(0)
        list_01.append(next_pos)
        list_10.append(2 * next_pos)
        list_11.append(3 * next_pos)

    tensor_11 = np.array(list_11, dtype=dtype)
    tensor_10 = np.array(list_10, dtype=dtype)
    tensor_01 = np.array(list_01, dtype=dtype)
    tensor_00 = np.array(list_00, dtype=dtype)
    tensor_00_inv = np.invert(tensor_11)
    tensor_01_inv = np.invert(tensor_10)
    tensor_10_inv = np.invert(tensor_01)
    index_bit = np.arange(0, num_bits, 2)

    if error_pat == "00":
        num_00, index_00 = count_pattern(orig_weight, tensor_00, tensor_11, index_bit)
        num_error_00 = int(error_rate*num_00)
        error00_randn_index = np.random.permutation(num_00)[:num_error_00]
        error00_indices = index_00[error00_randn_index, :]
        tensor10_index = tensor_10[(error00_indices[:, 1] / 2).astype(dtype)]
        tensor01_index = tensor_01[(error00_indices[:, 1] / 2).astype(dtype)]
        tensor11_index = tensor_11[(error00_indices[:, 1] / 2).astype(dtype)]
        if des_pat == "01":
            np.bitwise_or.at(weight, error00_indices[:, 0], tensor01_index)
        elif des_pat == "10":
            np.bitwise_or.at(weight, error00_indices[:, 0], tensor10_index)
        elif des_pat == "11":
            np.bitwise_or.at(weight, error00_indices[:, 0], tensor11_index)

    if error_pat == "01":
        num_01, index_01 = count_pattern(orig_weight, tensor_01, tensor_11, index_bit)
        num_error_01 = int(error_rate*num_01)
        error01_randn_index = np.random.permutation(num_01)[:num_error_01]
        error01_indices = index_01[error01_randn_index, :]
        tensor10_index = tensor_10[(error01_indices[:, 1] / 2).astype(dtype)]
        tensor00_inv_index = tensor_00_inv[(error01_indices[:, 1] / 2).astype(dtype)]
        tensor11_index = tensor_11[(error01_indices[:, 1] / 2).astype(dtype)]
        if des_pat == "00":
            np.bitwise_and.at(weight, error01_indices[:, 0], tensor00_inv_index)
        elif des_pat == "10":
            np.bitwise_xor.at(weight, error01_indices[:, 0], tensor11_index)
        elif des_pat == "11":
            np.bitwise_or.at(weight, error01_indices[:, 0], tensor11_index)

    if error_pat == "10":
        num_10, index_10 = count_pattern(orig_weight, tensor_10, tensor_11, index_bit)
        num_error_10 = int(error_rate*num_10)
        error10_randn_index = np.random.permutation(num_10)[:num_error_10]
        error10_indices = index_10[error10_randn_index, :]
        tensor00_index = tensor_00[(error10_indices[:, 1] / 2).astype(dtype)]
        tensor01_index = tensor_01[(error10_indices[:, 1] / 2).astype(dtype)]
        tensor11_index = tensor_11[(error10_indices[:, 1] / 2).astype(dtype)]
        tensor00_inv_index = tensor_00_inv[(error10_indices[:, 1] / 2).astype(dtype)]
        if des_pat == "01":
            np.bitwise_xor.at(weight, error10_indices[:, 0], tensor11_index)
        elif des_pat == "00":
            np.bitwise_and.at(weight, error10_indices[:, 0], tensor00_inv_index)
        elif des_pat == "11":
            np.bitwise_or.at(weight, error10_indices[:, 0], tensor11_index)

    if error_pat == "11":
        num_11, index_11 = count_pattern(orig_weight, tensor_11, tensor_11, index_bit)
        num_error_11 = int(error_rate*num_11)
        error11_randn_index = np.random.permutation(num_11)[:num_error_11]
        error11_indices = index_11[error11_randn_index, :]
        tensor10_index = tensor_10[(error11_indices[:, 1] / 2).astype(dtype)]
        tensor01_index = tensor_01[(error11_indices[:, 1] / 2).astype(dtype)]
        tensor00_index = tensor_00[(error11_indices[:, 1] / 2).astype(dtype)]
        tensor01_inv_index = tensor_01_inv[(error11_indices[:, 1] / 2).astype(dtype)]
        tensor10_inv_index = tensor_10_inv[(error11_indices[:, 1] / 2).astype(dtype)]
        tensor00_inv_index = tensor_00_inv[(error11_indices[:, 1] / 2).astype(dtype)]
        if des_pat == "01":
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor01_inv_index)
        elif des_pat == "10":
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor10_inv_index)
        elif des_pat == "00":
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor00_inv_index)

    return weight

def count_pattern(weight, tensor, tensor_11, index_bit):
    indices = []
    num = 0
    for tensor_i, tensor_11_i, index_b in zip(tensor, tensor_11, index_bit):
        and_result = np.bitwise_and(tensor_11_i, weight)
        index = (and_result == tensor_i).nonzero()[0]
        bit_index = np.full(index.size, index_b)
        index_tensor = np.stack((index, bit_index))
        indices.append(index_tensor)
        num += index.size
    total_index = np.concatenate(indices, axis=1)
    total_index = total_index.transpose(1, 0)
    return num, total_index
