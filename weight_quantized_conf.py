import torch
import numpy as np

torch.set_printoptions(profile="full")

class weight_conf(object):
    def __init__(self, weight, num_bits, method):
        self.shape = weight.shape
        self.num_bits = num_bits
        self.weight = weight
        self.method = method

    def inject_error(self, mlc_error_rate):
        if self.num_bits == 8:
            dtype = np.int8
        else:
            dtype = np.uint16

        weight = self.weight
        orig_weight = np.copy(weight)

        list_11 = [3]
        list_10 = [2]
        list_01 = [1]
        list_00 = [0]
        for shift in range(2, self.num_bits, 2):
            next_pos = 2 ** (shift)
            list_11.append(3 * next_pos)
            list_10.append(2 * next_pos)
            list_01.append(1 * next_pos)
            list_00.append(0)
        tensor_11 = np.array(list_11, dtype=dtype)
        tensor_10 = np.array(list_10, dtype=dtype)
        tensor_01 = np.array(list_01, dtype=dtype)
        tensor_00 = np.array(list_00, dtype=dtype)
        if self.num_bits == 10:
            tensor_00_inv = np.invert(tensor_11) - 64512
            tensor_10_inv = np.invert(tensor_01) - 64512
        else:
            tensor_00_inv = np.invert(tensor_11)
            tensor_10_inv = np.invert(tensor_01)

        index_bit = np.arange(0, self.num_bits, 2)

        if self.method == 'proposed':
            if mlc_error_rate["error_level3"] is not None:
                # Flip 11 --> 10:
                num_11, index_11 = count(weight, tensor_11, tensor_11, index_bit, self.num_bits)
                error_rate3 = mlc_error_rate["error_level3"]
                num_error_11 = int(num_11 * error_rate3)
                error11_randn_index = np.random.permutation(num_11)[:num_error_11]
                error11_indices = index_11[error11_randn_index, :]
                tensor10_inv_index = tensor_10_inv[(error11_indices[:, 1] / 2).astype(dtype)]
                np.bitwise_and.at(weight, error11_indices[:, 0], tensor10_inv_index)

            if mlc_error_rate["error_level2"] is not None:

                # Flip 01 --> 11:
                num_01, index_01 = count(orig_weight, tensor_01, tensor_11, index_bit, self.num_bits)
                error_rate2 = mlc_error_rate["error_level2"]
                num_error_01 = int(num_01 * error_rate2)
                error01_randn_index = np.random.permutation(num_01)[:num_error_01]
                error01_indices = index_01[error01_randn_index, :]
                tensor11_index= tensor_11[(error01_indices[:, 1] / 2).astype(dtype)]
                np.bitwise_or.at(weight, error01_indices[:, 0], tensor11_index)

        # 00 , 01, 11, 10
        if self.method == 'baseline':
            if mlc_error_rate["error_level3"] is not None:
                # Flip 11 --> 10:
                num_11, index_11 = count(orig_weight, tensor_11, tensor_11, index_bit, self.num_bits)
                error_rate3 = mlc_error_rate["error_level3"]
                num_error_11 = int(num_11 * error_rate3)
                error11_randn_index = np.random.permutation(num_11)[:num_error_11]
                error11_indices = index_11[error11_randn_index, :]
                tensor10_inv_index = tensor_10_inv[(error11_indices[:, 1] / 2).astype(dtype)]
                np.bitwise_and.at(weight, error11_indices[:, 0], tensor10_inv_index)

            if mlc_error_rate["error_level2"] is not None:

                # Flip 01 --> 11:
                num_01, index_01 = count(orig_weight, tensor_01, tensor_11, index_bit, self.num_bits)
                error_rate2 = mlc_error_rate["error_level2"]
                num_error_01 = int(num_01 * error_rate2)
                error01_randn_index = np.random.permutation(num_01)[:num_error_01]
                error01_indices = index_01[error01_randn_index, :]
                tensor11_index = tensor_11[(error01_indices[:, 1] / 2).astype(dtype)]
                np.bitwise_or.at(weight, error01_indices[:, 0], tensor11_index)

        return weight

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

