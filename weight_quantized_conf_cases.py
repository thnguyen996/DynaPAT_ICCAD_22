import torch
import numpy as np

torch.set_printoptions(profile="full")

class weight_conf(object):
    def __init__(self, weight, weight_type, num_bits):
        self.shape = weight.shape
        self.num_bits = num_bits
        self.weight = weight.view(-1)
        self.weight_type = weight_type

    def inject_error(self, mlc_error_rate, case):
        # num_error_00, num_error_11 = num_error
        case = int(case)
        num_mlc = self.weight_type["MLC"]
        if self.num_bits == 16:
            dtype = np.uint16
        elif self.num_bits == 8:
            dtype = np.uint8
        weight = self.weight
        orig_weight = weight.clone()
        weight = weight.cpu().numpy().astype(dtype)
        orig_weight = orig_weight.cpu().numpy().astype(dtype)

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
        tensor_00_inv = np.invert(tensor_11)
        tensor_10_inv = np.invert(tensor_01)
        tensor_01_inv = np.invert(tensor_10)

        index_bit = np.arange(0, self.num_bits, 2)


        error_rate2 = mlc_error_rate["error_level2"]
        error_rate3 = mlc_error_rate["error_level3"]
        if case == 1:

            # Flip 00 --> 01:
            num_00, index_00 = count(weight, tensor_00, tensor_11, index_bit, self.num_bits)
            num_error_00 = int(num_00 * error_rate3)
            error00_randn_index = np.random.permutation(num_00)[:num_error_00]
            error00_indices = index_00[error00_randn_index, :]

            tensor01_index = tensor_01[(error00_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_or.at(weight, error00_indices[:, 0], tensor01_index)

            # Flip 10 --> 00:
            num_10, index_10 = count(orig_weight, tensor_10, tensor_11, index_bit, self.num_bits)
            num_error_10 = int(num_10 * error_rate2)
            error10_randn_index = np.random.permutation(num_10)[:num_error_10]
            error10_indices = index_10[error10_randn_index, :]
            tensor00_inv_index = tensor_00_inv[(error10_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_and.at(weight, error10_indices[:, 0], tensor00_inv_index)

        elif case == 2:
            # Flip 00 --> 01:
            num_00, index_00 = count(weight, tensor_00, tensor_11, index_bit, self.num_bits)
            num_error_00 = int(num_00 * error_rate3)
            error00_randn_index = np.random.permutation(num_00)[:num_error_00]
            error00_indices = index_00[error00_randn_index, :]

            tensor01_index = tensor_01[(error00_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_or.at(weight, error00_indices[:, 0], tensor01_index)

            # Flip 11 --> 00:
            num_11, index_11 = count(orig_weight, tensor_11, tensor_11, index_bit, self.num_bits)
            num_error_11 = int(num_11 * error_rate2)
            error11_randn_index = np.random.permutation(num_11)[:num_error_11]
            error11_indices = index_11[error11_randn_index, :]
            tensor00_inv_index = tensor_00_inv[(error11_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor00_inv_index)

        elif case == 3:
            # Flip 11 --> 10:
            num_11, index_11 = count(weight, tensor_11, tensor_11, index_bit, self.num_bits)
            num_error_11 = int(num_11 * error_rate3)
            error11_randn_index = np.random.permutation(num_11)[:num_error_11]
            error11_indices = index_11[error11_randn_index, :]

            tensor10_inv_index = tensor_10_inv[(error11_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor10_inv_index)

            # Flip 01 --> 11:
            num_01, index_01 = count(orig_weight, tensor_01, tensor_11, index_bit, self.num_bits)
            num_error_01 = int(num_01 * error_rate2)
            error01_randn_index = np.random.permutation(num_01)[:num_error_01]
            error01_indices = index_01[error01_randn_index, :]

            tensor11_index= tensor_11[(error01_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_or.at(weight, error01_indices[:, 0], tensor11_index)

        elif case == 4:
            # Flip 11 --> 10:
            num_11, index_11 = count(weight, tensor_11, tensor_11, index_bit, self.num_bits)
            num_error_11 = int(num_11 * error_rate3)
            error11_randn_index = np.random.permutation(num_11)[:num_error_11]
            error11_indices = index_11[error11_randn_index, :]

            tensor10_inv_index = tensor_01_inv[(error11_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor10_inv_index)

            # Flip 00 --> 11:
            num_00, index_00 = count(orig_weight, tensor_00, tensor_11, index_bit, self.num_bits)
            num_error_00 = int(num_00 * error_rate2)
            error00_randn_index = np.random.permutation(num_00)[:num_error_00]
            error00_indices = index_00[error00_randn_index, :]

            tensor11_index = tensor_11[(error00_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_or.at(weight, error00_indices[:, 0], tensor11_index)

        elif case == 5:
            # Flip 11 --> 00:
            num_11, index_11 = count(weight, tensor_11, tensor_11, index_bit, self.num_bits)
            num_error_11 = int(num_11 * error_rate3)
            error11_randn_index = np.random.permutation(num_11)[:num_error_11]
            error11_indices = index_11[error11_randn_index, :]
            tensor00_inv_index = tensor_00_inv[(error11_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor00_inv_index)

            # Flip 01 --> 11:
            num_01, index_01 = count(orig_weight, tensor_01, tensor_11, index_bit, self.num_bits)
            num_error_01 = int(num_01 * error_rate2)
            error01_randn_index = np.random.permutation(num_01)[:num_error_01]
            error01_indices = index_01[error01_randn_index, :]

            tensor11_index = tensor_11[(error01_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_or.at(weight, error01_indices[:, 0], tensor11_index)

        elif case == 6:
            # Flip 11 --> 00:
            num_11, index_11 = count(weight, tensor_11, tensor_11, index_bit, self.num_bits)
            num_error_11 = int(num_11 * error_rate3)
            error11_randn_index = np.random.permutation(num_11)[:num_error_11]
            error11_indices = index_11[error11_randn_index, :]

            tensor00_inv_index = tensor_00_inv[(error11_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor00_inv_index)

            # Flip 10 --> 11:
            num_10, index_10 = count(orig_weight, tensor_10, tensor_11, index_bit, self.num_bits)
            num_error_10 = int(num_10 * error_rate2)
            error10_randn_index = np.random.permutation(num_10)[:num_error_10]
            error10_indices = index_10[error10_randn_index, :]

            tensor11_index = tensor_11[(error10_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_or.at(weight, error10_indices[:, 0], tensor11_index)

        elif case == 7:
            # Flip 11 --> 01:
            num_11, index_11 = count(weight, tensor_11, tensor_11, index_bit, self.num_bits)
            num_error_11 = int(num_11 * error_rate3)
            error11_randn_index = np.random.permutation(num_11)[:num_error_11]
            error11_indices = index_11[error11_randn_index, :]

            tensor01_inv_index = tensor_01_inv[(error11_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor01_inv_index)

            # Flip 00 --> 11:
            num_00, index_00 = count(orig_weight, tensor_00, tensor_11, index_bit, self.num_bits)
            num_error_00 = int(num_00 * error_rate2)
            error00_randn_index = np.random.permutation(num_00)[:num_error_00]
            error00_indices = index_00[error00_randn_index, :]

            tensor11_index = tensor_11[(error00_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_or.at(weight, error00_indices[:, 0], tensor11_index)

        elif case == 8:
            # Flip 11 --> 01:
            num_11, index_11 = count(weight, tensor_11, tensor_11, index_bit, self.num_bits)
            num_error_11 = int(num_11 * error_rate3)
            error11_randn_index = np.random.permutation(num_11)[:num_error_11]
            error11_indices = index_11[error11_randn_index, :]
            tensor01_inv_index = tensor_01_inv[(error11_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_and.at(weight, error11_indices[:, 0], tensor01_inv_index)
            # Flip 01 --> 11:
            num_01, index_01 = count(orig_weight, tensor_01, tensor_11, index_bit, self.num_bits)
            num_error_01 = int(num_01 * error_rate2)
            error01_randn_index = np.random.permutation(num_01)[:num_error_01]
            error01_indices = index_01[error01_randn_index, :]
            tensor11_index = tensor_11[(error01_indices[:, 1] / 2).astype(dtype)]
            np.bitwise_or.at(weight, error01_indices[:, 0], tensor11_index)
        else:
            print("You need to have a case")

        weight_float = weight.astype(np.float32)

        weight_float = torch.from_numpy(weight_float).to(self.weight.device)

        return weight_float

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

