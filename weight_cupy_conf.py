import torch
import cupy as cp
import numpy as np

torch.set_printoptions(profile="full")

class weight_conf(object):
    def __init__(self, weight, weight_type):
        self.shape = weight.shape
        self.device = weight.device
        self.weight = weight.view(-1).cpu()
        self.weight_type = weight_type

    def inject_error(self, mlc_error_rate, flip=True):
        num_mlc = self.weight_type["MLC"]
        weight = cp.asarray(self.weight)
        weight = weight.view(cp.uint32)
        orig_weight = cp.copy(weight)

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
        index_bit = cp.arange(0, 32, 2)
        tensor_11 = cp.asarray(list_11, dtype=cp.uint32)
        tensor_10 = cp.asarray(list_10, dtype=cp.uint32)
        tensor_01 = cp.asarray(list_01, dtype=cp.uint32)
        tensor_00 = cp.asarray(list_00, dtype=cp.uint32)
        tensor_00_inv = cp.invert(tensor_11)
        tensor_01_inv = cp.invert(tensor_10)
        tensor_10_inv = cp.invert(tensor_01)

        if num_mlc != 32:
            tensor_11 = tensor_11[: int(num_mlc / 2)]
            tensor_10 = tensor_10[: int(num_mlc / 2)]
            tensor_01 = tensor_01[: int(num_mlc / 2)]
            tensor_00 = tensor_00[: int(num_mlc / 2)]
            tensor_00_inv = tensor_00_inv[: int(num_mlc / 2)]
            tensor_01_inv = tensor_01_inv[: int(num_mlc / 2)]
            tensor_10_inv = tensor_10_inv[: int(num_mlc / 2)]
            index_bit = index_bit[: int(num_mlc / 2)]

        # Encoding: 
        if flip:
            # reshape weight matrix: 128x128
            weight, check_bits = flip_encode(weight, tensor_11, tensor_10, tensor_01, index_bit)

        # Error level 3: 01 --> 00
        if mlc_error_rate["error_lv3"] is not None:
            num_01, index_01 = count_pattern(weight, tensor_01, tensor_11, index_bit)
            error_rate = mlc_error_rate["error_lv3"]
            num_error_01 = int(error_rate*num_01)
            error01_randn_index = cp.random.permutation(num_01)[:num_error_01]
            error01_indices = index_01[error01_randn_index, :]
            tensor10_index = tensor_10[(error01_indices[:, 1] / 2).astype(cp.uint32)]
            tensor00_inv_index = tensor_00_inv[(error01_indices[:, 1] / 2).astype(cp.uint32)]
            tensor11_index = tensor_11[(error01_indices[:, 1] / 2).astype(cp.uint32)]
            weight_np = cp.asnumpy(weight)
            error01_indices_np = cp.asnumpy(error01_indices)

            tensor00_inv_index_np = cp.asnumpy(tensor00_inv_index)
            np.bitwise_and.at(weight_np, error01_indices_np[:, 0], tensor00_inv_index_np)
            weight = cp.asarray(weight_np)

        # Error level 2: 11 --> 01:
        if mlc_error_rate["error_lv2"] is not None:
            num_11, index_11 = count_pattern(orig_weight, tensor_11, tensor_11, index_bit)
            error_rate = mlc_error_rate["error_lv2"]
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
            # Flip 11 --> 01:

            tensor01_inv_index_np = cp.asnumpy(tensor01_inv_index)
            np.bitwise_and.at(weight_np, error11_indices_np[:, 0], tensor01_inv_index_np)
            weight = cp.asarray(weight_np)

        # Decoding:
        if flip:
            weight = flip_decode(weight, tensor_11, tensor_10, tensor_01, index_bit, check_bits)
        # Convert to 32 bit floating point pytorch tensor
        weight_float = weight.view(cp.float32)
        weight_error = torch.tensor(weight_float)
        # weight_float[(weight_float < 0.).nonzero()] = weight_float[(weight_float < 0.).nonzero()] + 2**16

        return weight_error.to(self.device)

def flip_encode(weight, tensor_11, tensor_10, tensor_01, index_bit, size=128):
    dim_1_shape = int(weight.shape[0]/4)
    if (dim_1_shape % size) == 0:
        weight = weight.reshape(int(dim_1_shape/size), size, 4)
        num_01, index_01 = count_pattern(weight, tensor_01, tensor_11, index_bit)
        index_01 = count_pattern_muldims(weight, tensor_01, tensor_11, index_bit)
        index_10 = count_pattern_muldims(weight, tensor_10, tensor_11, index_bit)
        check_bits = (index_01 > index_10)
        zeros = cp.zeros_like(weight)
        for index, tensor_11_i in enumerate(tensor_11):
            con = cp.repeat(cp.expand_dims(check_bits[:, :, index], 1), weight.shape[1], axis=1)
            tensor_11= cp.full_like(zeros, tensor_11_i)
            mask = cp.where(con, tensor_11, zeros)
            weight = cp.bitwise_xor(weight, mask)
        weight = weight.flatten()
    else:
        weight = weight.reshape(dim_1_shape, 4)
        weight_div_part = weight[:(dim_1_shape - (dim_1_shape % size)), :]
        weight_div_part = weight_div_part.reshape(int(weight_div_part.shape[0]/size), size, 4)
        weight_remain = cp.expand_dims(weight[(dim_1_shape - (dim_1_shape % size)):, :], 0)
        index_01_div = count_pattern_muldims(weight_div_part, tensor_01, tensor_11, index_bit)
        index_01_remain = count_pattern_muldims(weight_remain, tensor_01, tensor_11, index_bit)
        index_10_div = count_pattern_muldims(weight_div_part, tensor_10, tensor_11, index_bit)
        index_10_remain = count_pattern_muldims(weight_remain, tensor_10, tensor_11, index_bit)
        check_div = (index_01_div > index_10_div)
        check_remain = (index_01_remain > index_10_remain)

        zeros_remain = cp.zeros_like(weight_remain)
        zeros_div = cp.zeros_like(weight_div_part)
        for index, tensor_11_i in enumerate(tensor_11):
            con_remain = cp.repeat(cp.expand_dims(check_remain[:, :, index], 1), weight_remain.shape[1], axis=1)
            con_div = cp.repeat(cp.expand_dims(check_div[:, :, index], 1), weight_div_part.shape[1], axis=1)
            tensor_11_remain = cp.full_like(zeros_remain, tensor_11_i)
            tensor_11_div = cp.full_like(zeros_div, tensor_11_i)
            mask_remain = cp.where(con_remain, tensor_11_remain, zeros_remain)
            mask_div = cp.where(con_div, tensor_11_div, zeros_div)
            weight_remain = cp.bitwise_xor(weight_remain, mask_remain)
            weight_div_part = cp.bitwise_xor(weight_div_part, mask_div)
        # Merge weight_div and remain parts
        weight_div_part = weight_div_part.reshape(-1, 4)
        weight_remain = weight_remain.reshape(-1, 4)
        weight = cp.concatenate((weight_div_part, weight_remain), axis=0).flatten()
        check_bits = (check_div, check_remain)
    return weight, check_bits

def flip_decode(weight, tensor_11, tensor_10, tensor_01, index_bit, check_bits, size=128):
    dim_1_shape = int(weight.shape[0]/4)
    if (dim_1_shape % size) == 0:
        weight = weight.reshape(int(dim_1_shape/size), size, 4)
        num_01, index_01 = count_pattern(weight, tensor_01, tensor_11, index_bit)
        index_01 = count_pattern_muldims(weight, tensor_01, tensor_11, index_bit)
        index_10 = count_pattern_muldims(weight, tensor_10, tensor_11, index_bit)
        zeros = cp.zeros_like(weight)
        for index, tensor_11_i in enumerate(tensor_11):
            con = cp.repeat(cp.expand_dims(check_bits[:, :, index], 1), weight.shape[1], axis=1)
            tensor_11= cp.full_like(zeros, tensor_11_i)
            mask = cp.where(con, tensor_11, zeros)
            weight = cp.bitwise_xor(weight, mask)
        weight = weight.flatten()
    else:
        weight = weight.reshape(dim_1_shape, 4)
        weight_div_part = weight[:(dim_1_shape - (dim_1_shape % size)), :]
        weight_div_part = weight_div_part.reshape(int(weight_div_part.shape[0]/size), size, 4)
        weight_remain = cp.expand_dims(weight[(dim_1_shape - (dim_1_shape % size)):, :], 0)
        check_div, check_remain = check_bits

        zeros_remain = cp.zeros_like(weight_remain)
        zeros_div = cp.zeros_like(weight_div_part)
        for index, tensor_11_i in enumerate(tensor_11):
            con_remain = cp.repeat(cp.expand_dims(check_remain[:, :, index], 0), weight_remain.shape[1], axis=1)
            con_div = cp.repeat(cp.expand_dims(check_div[:, :, index], 1), weight_div_part.shape[1], axis=1)
            tensor_11_remain = cp.full_like(zeros_remain, tensor_11_i)
            tensor_11_div = cp.full_like(zeros_div, tensor_11_i)
            mask_remain = cp.where(con_remain, tensor_11_remain, zeros_remain)
            mask_div = cp.where(con_div, tensor_11_div, zeros_div)
            weight_remain = cp.bitwise_xor(weight_remain, mask_remain)
            weight_div_part = cp.bitwise_xor(weight_div_part, mask_div)
        # Merge weight_div and remain parts
        weight_div_part = weight_div_part.reshape(-1, 4)
        weight_remain = weight_remain.reshape(-1, 4)
        weight = cp.concatenate((weight_div_part, weight_remain), axis=0).flatten()
    return weight

def count_pattern_muldims(weight, tensor, tensor_11, index_bit):
    and_result = cp.bitwise_and(tensor_11[0], weight)
    index = (and_result == tensor[0]).nonzero()
    if index[0].size == 0:
        count_weight = cp.empty((weight.shape[0], 4, 16), dtype=cp.uint32)
    else:
        split_index = cp.unique(index[0], return_index=True)[1].tolist()
        list_array = cp.split(index[2], split_index)
        list_array.pop(0)
        count_weight = cp.empty((len(list_array), 4, 16), dtype=cp.uint32)
    # count_weight = cp.unique(list_array[0], return_counts=True)[1]
        for index, array in enumerate(list_array):
            count_weight[index, :, 0] = cp.unique(array, return_counts=True)[1]

    # num_array = cp.unique(index, return_counts=True)[1]
    # num_array = cp.expand_dims(num_array, 1).transpose(1, 0)
    for tensor_i, tensor_11_i, index_b in zip(tensor[1:], tensor_11[1:], index_bit[1:]):
        and_result = cp.bitwise_and(tensor_11_i, weight)
        index = (and_result == tensor_i).nonzero()
        index_b = int(index_b/2)
        if index[0].size == 0:
            count_weight[:, :, index_b] = cp.zeros((count_weight.shape[0], 4))
        else:
            # num = cp.unique(index[0], return_counts=True)
            # num = cp.expand_dims(num[1], 0)
            split_index = cp.unique(index[0], return_index=True)[1].tolist()
            list_array = cp.split(index[2], split_index)
            list_array.pop(0)
            for index, array in enumerate(list_array):
                count_index = cp.unique(array, return_counts=True)
                # if cp.unique(array, return_counts=True)[1].size < 4:
                #     import pdb; pdb.set_trace()
                count_weight[index, count_index[0], index_b] = count_index[1]
    # NOTE: The num_array is in reversed order, need to do reverse index
    return count_weight

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
