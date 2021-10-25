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

    def inject_error(self, weight, mlc_error_rate, tlc_error_rate=False):
        num_mlc = self.weight_type["MLC"]
        weight = cp.asarray(weight)
        weight = weight.view(cp.uint32)
        orig_weight = cp.copy(weight)

        # create tensor 11 and indices

        if mlc_error_rate["error_11"] is not None:
            list_11 = [3]
            list_10 = [1]
            for shift in range(2, 32, 2):
                next_pos = 2 ** (shift)
                list_11.append(3 * next_pos)
                list_10.append(next_pos)
            tensor_11 = cp.asarray(list_11, dtype=cp.uint32)
            tensor_10 = cp.invert(cp.asarray(list_10, dtype=cp.uint32))
            num_11 = 0
            indices_11 = []
            index_bit = cp.arange(0, 32, 2)
            if num_mlc != 32:
                tensor_11 = tensor_11[: int(num_mlc / 2)]
                tensor_10 = tensor_10[: int(num_mlc / 2)]
                index_bit = index_bit[: int(num_mlc / 2)]

            # count number of 11 and take index
            for tensor_11_i, index_b in zip(tensor_11, index_bit):
                and_result = cp.bitwise_and(tensor_11_i, weight)
                index_11 = (and_result == tensor_11_i).nonzero()[0]
                bit_index = cp.full(index_11.size, index_b)
                index_tensor = cp.stack((index_11, bit_index))
                indices_11.append(index_tensor)
                num_11 += index_11.size

            total_index_11 = cp.concatenate(indices_11, axis=1)
            total_index_11 = total_index_11.transpose(1, 0)

            error_rate_11 = mlc_error_rate["error_11"]
            num_error_11 = int(num_11 * error_rate_11)
            error11_randn_index = cp.random.permutation(num_11)[:num_error_11]
            error11_indices = total_index_11[error11_randn_index, :]

            tensor10 = tensor_10[(error11_indices[:, 1] / 2).astype(cp.uint32)]
            # Flip 11 --> 10:
            # Got to move to numpy to use bitwise_.at operation: Feel free to contribute
            weight_np = cp.asnumpy(weight)
            tensor10_np = cp.asnumpy(tensor10)
            error11_indices = cp.asnumpy(error11_indices)
            np.bitwise_and.at(weight_np, error11_indices[:, 0], tensor10_np)
            weight = cp.asarray(weight_np)

        if mlc_error_rate["error_10"] is not None:
            # count number of 10 and take index
            list_10 = [2]
            list_11 = [3]
            for shift in range(2, 32, 2):
                next_pos = 2 ** (shift)
                list_10.append(2 * next_pos)
                list_11.append(3 * next_pos)
            tensor_10 = cp.asarray(list_10, dtype=cp.uint32)
            tensor_11 = cp.asarray(list_11, dtype=cp.uint32)
            num_10 = 0
            indices_10 = []
            index_bit = cp.arange(0, 32, 2)
            if num_mlc != 32:
                tensor_11 = tensor_11[: int(num_mlc / 2)]
                tensor_10 = tensor_10[: int(num_mlc / 2)]
                index_bit = index_bit[: int(num_mlc / 2)]

            for tensor_10_i, tensor_11_i, index_b in zip(
                tensor_10, tensor_11, index_bit
            ):
                and_result = cp.bitwise_and(tensor_11_i, orig_weight)
                index_10 = (and_result == tensor_10_i).nonzero()[0]
                bit_index = cp.full(index_10.size, index_b)
                index_tensor = cp.stack((index_10, bit_index))
                indices_10.append(index_tensor)
                num_10 += index_10.size
            total_index_10 = cp.concatenate(indices_10, axis=1)
            total_index_10 = total_index_10.transpose(1, 0)

            error_rate_10 = mlc_error_rate["error_10"]
            num_error_10 = int(num_10 * error_rate_10)
            error10_randn_index = cp.random.permutation(num_10)[:num_error_10]
            error10_indices = total_index_10[error10_randn_index, :]

            tensor11 = tensor_11[(error10_indices[:, 1] / 2).astype(cp.uint32)]
            weight_np = cp.asnumpy(weight)
            tensor11_np = cp.asnumpy(tensor11)
            error10_indices = cp.asnumpy(error10_indices)
            np.bitwise_or.at(weight_np, error10_indices[:, 0], tensor11_np)
            weight = cp.asarray(weight_np)

        # Convert to 32 bit floating point pytorch tensor
        weight_float = weight.view(cp.float32)
        weight_error = torch.tensor(weight_float)
        # weight_float[(weight_float < 0.).nonzero()] = weight_float[(weight_float < 0.).nonzero()] + 2**16

        return num_error_11, num_11, num_error_10, num_10


