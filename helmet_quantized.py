import torch
import numpy as np


def helmet_en(weight, num_bits):
    if num_bits == 8:
        dtype = np.int8
    elif num_bits == 16:
        dtype = np.int16
    # reshape to memristor length (128*128)
    weight = weight.reshape(int(weight.numel() / 2), -1)
    orig_weight = weight.clone()
    weight = weight.cpu().numpy().astype(dtype)
    orig_weight = orig_weight.cpu().numpy().astype(dtype)
    # Counts 01, 11, 00, 10
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

    inv_weight = np.invert(weight)
    rot_weight = circshift(weight, num_bits)
    ir_weight = circshift(inv_weight, num_bits)

    num_11_orig = count_11(weight, tensor_11, tensor_11, num_bits)
    num_11_inv = count_11(inv_weight, tensor_11, tensor_11, num_bits)
    num_11_rot = count_11(rot_weight, tensor_11, tensor_11, num_bits)
    num_11_ir  = count_11(ir_weight, tensor_11,  tensor_11, num_bits)

    num_11_orig = np.sum(num_11_orig, axis=1)
    num_11_inv = np.sum(num_11_inv, axis=1)
    num_11_rot = np.sum(num_11_rot, axis=1)
    num_11_ir = np.sum(num_11_ir, axis=1)

    total_11 = np.stack((num_11_orig, num_11_inv, num_11_rot, num_11_ir))
    min_case = np.argmin(total_11, axis=0)

    weight[(min_case == 1).nonzero()[0], :] = inv_weight[(min_case == 1).nonzero()[0], :]
    weight[(min_case == 2).nonzero()[0], :] = rot_weight[(min_case == 2).nonzero()[0], :]
    weight[(min_case == 3).nonzero()[0], :] = ir_weight[(min_case == 3).nonzero()[0], :]

    if num_bits == 16:
        weight = weight.astype(np.uint16)
    weight_torch = torch.tensor(weight.astype(np.float32), device="cuda")

    return torch.flatten(weight_torch)

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

def circshift_left(weight):
    weight_np = weight.view(np.uint16)
    save_bit = np.right_shift(weight_np, 15)
    rot_bit = np.left_shift(weight_np, 1)
    rot_weight = np.bitwise_or(save_bit, rot_bit)
    return rot_weight.view(np.int16)

def helmet_de(weight, fc):
    weight = weight.reshape(int(weight.numel() / 16), 16)
    weight[(fc == 1).nonzero().squeeze(1), :] = torch.bitwise_not(weight[(fc == 1).nonzero().squeeze(1), :])
    weight[(fc == 2).nonzero().squeeze(1), :] = circshift_left(weight[(fc == 2).nonzero().squeeze(1), :])
    weight[(fc == 3).nonzero().squeeze(1), :] = torch.bitwise_not(circshift_left(weight[(fc == 3).nonzero().squeeze(1), :]))

    return weight.flatten()

def count_11(weight, tensor_01, tensor_11, num_bits):
    index_bit = np.arange(0, num_bits, 2)
    num_01 = 0
    indices_01 = []
    num_01 = np.zeros_like(weight)
    for tensor_01_i, tensor_11_i, index_b in zip(tensor_01, tensor_11, index_bit):
        and_result = np.bitwise_and(tensor_11_i, weight)
        index_01 = (and_result == tensor_01_i).nonzero()
        num_01[index_01[0], index_01[1]] += 1

        # bit_index = np.full_like(index_01, index_b)
        # index_tensor = np.stack((index_01, bit_index))
        # indices_01.append(index_tensor)
        # num_01 += index_01.size
    # total_index_01 = np.concatenate(indices_01, axis=1)
    # indices = np.unique(total_index_01[0, :], return_counts=True)
    # zeros = np.zeros(weight.size)
    # zeros[indices[0]] = indices[1]
    return num_01

def helmet_inject_error(weight_type, weight, mlc_error_rate, tlc_error_rate=False):
    num_mlc = weight_type["MLC"]
    weight = weight.type(torch.int16)
    orig_weight = weight.clone()

    # create tensor 11 and indices
    if mlc_error_rate["error_11"] is not None:
        list_11 = [3]
        list_10 = [1]
        for shift in range(2, 16, 2):
            next_pos = 2 ** (shift)
            list_11.append(3 * next_pos)
            list_10.append(next_pos)
        tensor_11 = torch.tensor(
            list_11, dtype=torch.int16, device=weight.device
        )
        tensor_10 = torch.bitwise_not(
            torch.tensor(list_10, dtype=torch.int16, device=weight.device)
        )
        num_11 = 0
        indices_11 = []
        index_bit = torch.arange(0, 16, 2)
        if num_mlc != 16:
            tensor_11 = tensor_11[:int(num_mlc/2)]
            tensor_10 = tensor_10[:int(num_mlc/2)]
            index_bit = index_bit[:int(num_mlc/2)]

        # count number of 11 and take index
        for tensor_11_i, index_b in zip(tensor_11, index_bit):
            and_result = torch.bitwise_and(tensor_11_i, weight)
            index_11 = (and_result == tensor_11_i).nonzero()
            bit_index = torch.zeros_like(index_11).fill_(index_b)
            index_tensor = torch.stack((index_11, bit_index))
            indices_11.append(index_tensor)
            num_11 += index_11.numel()
        total_index_11 = torch.cat(indices_11, dim=1)
        total_index_11.squeeze_(2).transpose_(1, 0)

        error_rate_11 = mlc_error_rate["error_11"]
        num_error_11 = int(num_11 * error_rate_11)
        error11_randn_index = torch.randperm(num_11)[:num_error_11]
        error11_indices = total_index_11[error11_randn_index, :]

        tensor10 = tensor_10[(error11_indices[:, 1] / 2).type(torch.long)]
        # Flip 11 --> 10:
        # Got to move to numpy to use bitwise_.at operation: Feel free to contribute
        weight_np = weight.cpu().numpy()
        tensor10_np = tensor10.cpu().numpy()
        error11_indices = error11_indices.cpu().numpy()
        np.bitwise_and.at(weight_np, error11_indices[:, 0], tensor10_np)
        weight = torch.from_numpy(weight_np).to(weight.device)

        # weight[error11_indices[:, 0]] = torch.bitwise_and(
        #     weight[error11_indices[:, 0]], tensor10
        # )

    if mlc_error_rate["error_10"] is not None:
        # count number of 10 and take index
        list_10 = [2]
        list_11 = [3]
        for shift in range(2, 16, 2):
            next_pos = 2 ** (shift)
            list_10.append(2 * next_pos)
            list_11.append(3 * next_pos)
        tensor_10 = torch.tensor(
            list_10, dtype=torch.int16, device=weight.device
        )
        tensor_11 = torch.tensor(
            list_11, dtype=torch.int16, device=weight.device
        )
        num_10 = 0
        indices_10 = []
        index_bit = torch.arange(0, 16, 2)
        if num_mlc != 16:
            tensor_11 = tensor_11[:int(num_mlc/2)]
            tensor_10 = tensor_10[:int(num_mlc/2)]
            index_bit = index_bit[:int(num_mlc/2)]

        for tensor_10_i, tensor_11_i, index_b in zip(tensor_10, tensor_11, index_bit):
            and_result = torch.bitwise_and(tensor_11_i, orig_weight)
            index_10 = (and_result == tensor_10_i).nonzero()
            bit_index = torch.zeros_like(index_10).fill_(index_b)
            index_tensor = torch.stack((index_10, bit_index))
            indices_10.append(index_tensor)
            num_10 += index_10.numel()
        total_index_10 = torch.cat(indices_10, dim=1)
        total_index_10.squeeze_(2).transpose_(1, 0)

        error_rate_10 = mlc_error_rate["error_10"]
        num_error_10 = int(num_10 * error_rate_10)
        error10_randn_index = torch.randperm(num_10)[:num_error_10]
        error10_indices = total_index_10[error10_randn_index, :]

        tensor11 = tensor_11[(error10_indices[:, 1] / 2).type(torch.long)]
        # Flip 10 --> 11:
        # weight[error10_indices[:, 0]] = torch.bitwise_or(
        #     weight[error10_indices[:, 0]], tensor11
        # )
        weight_np = weight.cpu().numpy()
        tensor11_np = tensor11.cpu().numpy()
        error10_indices = error10_indices.cpu().numpy()
        np.bitwise_or.at(weight_np, error10_indices[:, 0], tensor11_np)
        weight = torch.from_numpy(weight_np).to(weight.device)

# Convert to 16 bit unsigned
        # weight_float = weight.type(torch.float16)
        # weight_float[(weight_float < 0.).nonzero()] = weight_float[(weight_float < 0.).nonzero()] + 2**16
        weight_float = weight

        return weight_float
