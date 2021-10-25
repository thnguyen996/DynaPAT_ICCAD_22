import torch
torch.set_printoptions(profile="full")


class weight_conf(object):
    def __init__(self, weight, weight_type, level):
        self.weight = weight
        self.weight_type = weight_type
        self.level = level

    # Return a flatten binary weight:

    def weight_to_binary(self):
        weight_binary = float2bit(self.weight, 8, 23, 127.0)
        weight_flatten = weight_binary.reshape(-1, 32)
        return weight_flatten

    def inject_error(self, weight_flatten, mlc_error_rate, tlc_error_rate=False):
        # Convert to MLC:
        num_mlc = self.weight_type["MLC"]
        if num_mlc == 32:
            mlc_bit = weight_flatten[:, 0:num_mlc]
        else:
            mlc_bit = weight_flatten[:, 9 : 9 + num_mlc]
        shape = mlc_bit.shape
        mlc_bit = torch.reshape(mlc_bit, (-1, int(num_mlc / 2), 2))
        num_cell = int(mlc_bit.numel() / 2)
        orig_mlc_bit = mlc_bit.clone()

        # Count numbers of 01 pattern
        if mlc_error_rate["error_11"] is not None:
            sum_2_bit = mlc_bit.sum(dim=2)
            indices = torch.nonzero(
                sum_2_bit == torch.tensor([2.0], device=mlc_bit.device), as_tuple=False
            )
            num_11 = indices[:, 1].numel()

            # Flip 11 --> 10
            error_rate_11 = mlc_error_rate["error_11"]
            num_error_11 = int(error_rate_11 * num_11)
            random_pos11 = torch.randperm(num_11)[0:num_error_11]
            error11_indices = indices[random_pos11]
            tensor10 = torch.tensor([1.0, 0.0], device=mlc_bit.device)
            tensor10 = tensor10.repeat(num_error_11, 1)
            mlc_bit[error11_indices[:, 0], error11_indices[:, 1], :] = tensor10

        # Count the number of 10 and flip 10 --> 11:
        if mlc_error_rate["error_10"] is not None:
            sum_2_bit = orig_mlc_bit[:, :, 0] * 2 + orig_mlc_bit[:, :, 1]
            indices = torch.nonzero(
                sum_2_bit == torch.tensor([2.0], device=mlc_bit.device), as_tuple=False
            )
            num_10 = indices[:, 1].numel()
            error_rate_10 = mlc_error_rate["error_10"]
            num_error_10 = int(error_rate_10 * num_10)
            random_pos10 = torch.randperm(num_10)[0:num_error_10]
            error10_indices = indices[random_pos10]

            tensor11 = torch.tensor([1.0, 1.0], device=mlc_bit.device)
            tensor11 = tensor11.repeat(num_error_10, 1)
            mlc_bit[error10_indices[:, 0], error10_indices[:, 1], :] = tensor11
        mlc_bit = mlc_bit.reshape(shape)
        if num_mlc == 32:
            weight_flatten[:, 0:num_mlc] = mlc_bit
        else:
            weight_flatten[:, 9 : 9 + num_mlc] = mlc_bit
        orig_weight = bit2float(weight_flatten)

        return orig_weight

    # inject error in bit position
    def inject_bit(self, weight_flatten, bit_pos, error_rate):
        faulty_bits = weight_flatten[:, bit_pos].clone()
        shape = faulty_bits.shape
        faulty_bits_flatten = faulty_bits.view(-1)

        num_error = int(error_rate * faulty_bits.numel())
        if num_error != 0:
            error_index = torch.randperm(faulty_bits.numel())[:num_error]
            faulty_bits_flatten[error_index] = torch.bitwise_not(
                faulty_bits_flatten[error_index].type(torch.bool)
            ).type(torch.float32)

        # output = torch.logical_xor(faulty_bits_flatten.type(torch.bool), mask).type(torch.float32)
        weight_flatten[:, bit_pos] = faulty_bits_flatten.reshape(shape)
        orig_weight = bit2float(weight_flatten)

        return orig_weight

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)
