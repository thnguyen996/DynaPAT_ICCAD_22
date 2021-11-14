import torch
import numpy as np

def flipcy_en(weight, num_bits):
    if num_bits == 8:
        dtype = torch.uint8
        weight = weight.type(dtype).to("cuda")
    elif num_bits == 16:
        dtype = torch.int16
        weight = weight.type(dtype).to("cuda")
    # reshape to memristor length (128*128)
    weight = weight.reshape(int(weight.numel() / 1), 1)
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

    tensor_11 = torch.tensor(list_11, dtype=dtype, device=weight.device)
    tensor_10 = torch.tensor(list_10, dtype=dtype, device=weight.device)
    tensor_01 = torch.tensor(list_01, dtype=dtype, device=weight.device)
    tensor_00 = torch.tensor(list_00, dtype=dtype, device=weight.device)
    Flip = []
    Comp = []
    weight = weight.cpu().numpy()
    num_11 = count_11(weight, tensor_11, num_bits)
    num_10 = count(weight, tensor_10, tensor_11, num_bits)
    num_01 = count(weight, tensor_01, tensor_11, num_bits)
    num_00 = count(weight, tensor_00, tensor_11, num_bits)

    # take sum of 10 + 11, 00 + 10
    sum0111 = num_10 + num_11
    sum0010 = num_00 + num_10
    con_sum_index = (sum0111 > sum0010)
    con_index = (num_00 > num_10)
    con_index2 = (num_11 > num_01)

    case1_index = (con_sum_index & con_index).nonzero()[0]# if 01 + 11 > 00 + 10 and 00 > 10
    case2_index = (con_sum_index & np.invert(con_index)).nonzero()[0]# if 01 + 11 > 00 + 10 and 00 < 10
    case3_index = (np.invert(con_sum_index) & con_index2).nonzero()[0]# if 01 + 11 < 00 + 10 and 11 > 01

# Case 1: Flip and comp.
    tensor_11 = tensor_11.cpu().numpy()
    tensor_10 = tensor_10.cpu().numpy()
    tensor_01 = tensor_01.cpu().numpy()
    weight_case1 = np.invert(weight[case1_index, :])
    num_11_c1, index_11_c1 = count_orig(weight_case1, tensor_11, tensor_11, num_bits)
    num_01_c1, index_01_c1 = count_orig(weight_case1, tensor_01, tensor_11, num_bits)
    # 2's complements
    # FLip 01 --> 11
    c11 = tensor_11[(index_01_c1[:, 1] / 2).astype(np.int_)]
    np.bitwise_or.at(weight_case1[:, 0], index_01_c1[:, 0], c11)

    # FLip 11 --> 01
    c10 = np.invert(tensor_10[(index_11_c1[:, 1] / 2).astype(np.int_)])
    np.bitwise_and.at(weight_case1[:, 0], index_11_c1[:, 0], c10)
    weight[case1_index, :] = weight_case1

# Case2: Flip only
    weight[case2_index, :] = np.invert(weight[case2_index, :])
# Case3: Comp. only
    weight_case3 = weight[case3_index, :]
    num_11_c1, index_11_c1 = count_orig(weight_case3, tensor_11, tensor_11, num_bits)
    num_01_c1, index_01_c1 = count_orig(weight_case3, tensor_01, tensor_11, num_bits)
    # 2's complements
    # FLip 01 --> 11
    c11 = tensor_11[(index_01_c1[:, 1] / 2).astype(np.int_)]
    np.bitwise_or.at(weight_case3[:, 0], index_01_c1[:, 0], c11)

    # FLip 11 --> 01
    c10 = np.invert(tensor_10[(index_11_c1[:, 1] / 2).astype(np.int_)])
    np.bitwise_and.at(weight_case3[:, 0], index_11_c1[:, 0], c10)
    weight[case3_index, :] = weight_case3
    if num_bits == 16:
        weight = weight.astype(np.uint16)
    weight_torch = torch.tensor(weight.astype(np.float32), device="cuda")

    return torch.flatten(weight_torch)

def count_11(weight, tensor_11, num_bits):
    index_bit = np.arange(0, num_bits, 2)
    num_11 = 0
    indices_11 = []
    tensor_11 = tensor_11.cpu().numpy()
    for tensor_11_i, index_b in zip(tensor_11, index_bit):
        and_result = np.bitwise_and(tensor_11_i, weight)
        index_11 = (and_result == tensor_11_i).nonzero()[0]
        bit_index = np.full_like(index_11, index_b)
        bit_index = np.transpose(np.expand_dims(bit_index, 0), (1, 0))
        index_11 = np.transpose(np.expand_dims(index_11, 0), (1, 0))
        index_tensor = np.concatenate((index_11, bit_index), axis=1)
        indices_11.append(index_tensor)
        num_11 += index_11.shape[1]
    total_index_11 = np.concatenate(indices_11, axis=0)
    indices = np.unique(total_index_11[:, 0], return_counts=True)
    zeros = np.zeros(weight.size)
    zeros[indices[0]] = indices[1]
    return zeros


def count(weight, tensor_10, tensor_11, num_bits):
    index_bit = torch.arange(0, num_bits, 2)
    num_10 = 0
    indices_10 = []
    tensor_10 = tensor_10.cpu().numpy()
    tensor_11 = tensor_11.cpu().numpy()
    for tensor_10_i, tensor_11_i, index_b in zip(tensor_10, tensor_11, index_bit):
        and_result = np.bitwise_and(tensor_11_i, weight)
        index_10 = (and_result == tensor_10_i).nonzero()[0]
        bit_index = np.full_like(index_10, index_b)
        bit_index = np.transpose(np.expand_dims(bit_index, 0), (1, 0))
        index_10 = np.transpose(np.expand_dims(index_10, 0), (1, 0))
        index_tensor = np.concatenate((index_10, bit_index), axis=1)
        indices_10.append(index_tensor)
        num_10 += index_10.shape[1]
    total_index_10 = np.concatenate(indices_10, axis=0)
    indices = np.unique(total_index_10[:, 0], return_counts=True)
    zeros = np.zeros(weight.size)
    zeros[indices[0]] = indices[1]
    return zeros


def count_orig(weight, tensor_10, tensor_11, num_bits):
    index_bit = np.arange(0, num_bits, 2)
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
    total_index_10 = np.transpose(np.squeeze(total_index_10), (1, 0))
    return num_10, total_index_10

def inject_error(weight, num_error, mlc_error_rate, num_bits):
    num_error_11, num_error_01 = num_error
    if num_bits == 16:
        dtype = np.uint16
    elif num_bits == 8:
        dtype = np.uint8
    # weight = weight.cpu().numpy().astype(dtype)
    orig_weight = np.copy(weight)

    # create tensor 11 and indices

    if mlc_error_rate["error_11"] is not None:
        list_11 = [3]
        list_10 = [2]
        for shift in range(2, num_bits, 2):
            next_pos = 2 ** (shift)
            list_11.append(3 * next_pos)
            list_10.append(2 * next_pos)
        tensor_11 = np.array(list_11, dtype=dtype)
        tensor_10 = np.array(list_10, dtype=dtype)
        num_10 = 0
        indices_10 = []
        index_bit = np.arange(0, num_bits, 2)


        # Flip 11 --> 10:
        # Got to move to numpy to use bitwise_.at operation: Feel free to contribute
        # count number of 11 and take index
        num_11, total_index_11 = count_orig(weight, tensor_11, tensor_11, num_bits)
        error_rate_11 = mlc_error_rate["error_11"]
        # num_error_10 = int(num_10 * error_rate_10)
        error11_randn_index = np.random.permutation(num_11)[:num_error_11]
        error11_indices = total_index_11[error11_randn_index, :]

        tensor10 = tensor_10[(error11_indices[:, 1] / 2).astype(np.int_)]
        np.bitwise_or.at(weight, error11_indices[:, 0], tensor10)

    if mlc_error_rate["error_10"] is not None:
        # count number of 01 and take index
        list_01 = [1]
        list_11 = [3]
        for shift in range(2, num_bits, 2):
            next_pos = 2 ** (shift)
            list_01.append(1 * next_pos)
            list_11.append(3 * next_pos)
        tensor_01 = np.array( list_01, dtype=dtype)
        tensor_11 = np.array(list_11, dtype=dtype)
        num_01 = 0
        indices_01 = []
        index_bit = np.arange(0, num_bits, 2)

        # count number of 01 and take index
        num_01, total_index_01 = count_orig(orig_weight, tensor_01, tensor_11, num_bits)

        error_rate_01 = mlc_error_rate["error_10"]
        # num_error_01 = int(num_01 * error_rate_01)
        error01_randn_index = np.random.permutation(num_01)[:num_error_01]
        error01_indices = total_index_01[error01_randn_index, :]

        # Flip 01 --> 11:
        tensor11 = tensor_11[(error01_indices[:, 1] / 2).astype(np.int_)]
        np.bitwise_or.at(weight, error01_indices[:, 0], tensor11)

    if num_bits == 16:
        weight_float = weight.astype(np.uint16).astype(np.float32)
    elif num_bits == 8:
        weight_float = weight.astype(np.float32)
    weight_float = torch.from_numpy(weight_float).to("cuda")

# Convert to 16 bit unsigned
    # if self.num_bits == 16:
    #     weight_float[(weight_float < 0.).nonzero()] = weight_float[(weight_float < 0.).nonzero()] + 2**16
    return weight_float

def flipcy_de(weight, flip, comp):
    weight = weight.reshape(int(weight.numel() / 16), 16)
    # Counts 01, 11, 00, 10
    list_00 = [0]
    list_01 = [1]
    list_10 = [2]
    list_11 = [3]
    for shift in range(2, 16, 2):
        next_pos = 2 ** (shift)
        list_00.append(0)
        list_01.append(next_pos)
        list_10.append(2 * next_pos)
        list_11.append(3 * next_pos)
    tensor_11 = torch.tensor(list_11, dtype=torch.int16, device=weight.device)
    tensor_10 = torch.tensor(list_10, dtype=torch.int16, device=weight.device)
    tensor_01 = torch.tensor(list_01, dtype=torch.int16, device=weight.device)
    tensor_00 = torch.tensor(list_00, dtype=torch.int16, device=weight.device)

    flip = torch.tensor(flip)
    comp = torch.tensor(comp)
    fc = torch.stack((flip, comp)).squeeze().transpose(1, 0)
    fc10_index = torch.where((fc[:, 0] == torch.tensor([1])) & (fc[:, 1] == torch.tensor([0])))[0]
    fc01_index = torch.where((fc[:, 0] == torch.tensor([0])) & (fc[:, 1] == torch.tensor([1])))[0]
    fc11_index = torch.where((fc[:, 0] == torch.tensor([1])) & (fc[:, 1] == torch.tensor([1])))[0]

    # fc = 10 case
    weight[fc10_index, :] = torch.bitwise_not(weight[fc10_index, :])
    # fc = 01 case
    shape_fc01 = weight[fc01_index, :].shape
    weight_fc_01 = weight[fc01_index, :].flatten()
    orig_weight_fc_01 = weight_fc_01.clone()

    # Complement
    # FLip 01 --> 11
    _, index_01 = count_01(weight_fc_01, tensor_01, tensor_11)
    c11 = tensor_11[(index_01[:, 1] / 2).type(torch.long)]
    # Convert to numpy
    index_01_np = index_01.cpu().numpy()
    weight_fc_01_np = weight_fc_01.cpu().numpy()
    c11_np = c11.cpu().numpy()

    np.bitwise_or.at(weight_fc_01_np, index_01_np[:, 0], c11_np)
    weight_fc_01_torch = torch.tensor(weight_fc_01_np).to(weight.device)
    weight[fc01_index, :] = weight_fc_01_torch.reshape(shape_fc01)

    # FLip 11 --> 01
    _, index_11 = count_11(orig_weight_fc_01, tensor_11)
    c01 = torch.bitwise_not(tensor_10[(index_11[:, 1] / 2).type(torch.long)])
    # Convert to numpy
    index_11_np = index_11.cpu().numpy()
    weight_fc_01_np = weight_fc_01_torch.cpu().numpy()
    c01_np = c01.cpu().numpy()

    np.bitwise_and.at(weight_fc_01_np, index_11_np[:, 0], c01_np)
    weight_fc_01_torch = torch.tensor(weight_fc_01_np, device=weight.device)
    weight[fc01_index, :] = weight_fc_01_torch.reshape(shape_fc01)

    # fc = 11 case: FLip and complement
    shape_fc11 = weight[fc11_index, :].shape
    weight_fc_11 = weight[fc11_index, :].flatten()
    orig_weight_fc_11 = weight_fc_11.clone()

    # Complement
    # FLip 01 --> 11
    _, index_01 = count_01(weight_fc_11, tensor_01, tensor_11)
    c11 = tensor_11[(index_01[:, 1] / 2).type(torch.long)]
    # Convert to numpy
    index_01_np = index_01.cpu().numpy()
    weight_fc_11_np = weight_fc_11.cpu().numpy()
    c11_np = c11.cpu().numpy()

    np.bitwise_or.at(weight_fc_11_np, index_01_np[:, 0], c11_np)
    weight_fc_11_torch = torch.tensor(weight_fc_11_np, device=weight.device)
    weight[fc11_index, :] = weight_fc_11_torch.reshape(shape_fc11)

    # FLip 11 --> 01
    _, index_11 = count_11(orig_weight_fc_11, tensor_11)
    c01 = torch.bitwise_not(tensor_10[(index_11[:, 1] / 2).type(torch.long)])
    # Convert to numpy
    index_11_np = index_11.cpu().numpy()
    weight_fc_11_np = weight_fc_11_torch.cpu().numpy()
    c01_np = c01.cpu().numpy()

    np.bitwise_and.at(weight_fc_11_np, index_11_np[:, 0], c01_np)
    weight_fc_11_torch = torch.tensor(weight_fc_11_np, device=weight.device)
    weight[fc11_index, :] = weight_fc_11_torch.reshape(shape_fc11)

    # Invert
    weight[fc11_index, :] = torch.bitwise_not(weight[fc11_index, :])
    return weight.flatten()
