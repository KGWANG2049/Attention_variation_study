import torch
from torch import nn
from utils import ops
import math


class SelfAttention(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_points=2048, num_heads=8):
        super(SelfAttention, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        if q_out != v_out:
            raise ValueError('Please check the dimension of energy')
        # print('q_out, k_out, v_out are same')
        self.num_points = num_points
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.p_add = nn.Parameter(torch.ones(1, self.k_depth))  # q.shape == (1, D)

        self.linear_energy = nn.Linear(3, num_points * num_heads)
        self.linear_q = nn.Linear(3, q_out)
        self.linear_k = nn.Linear(3, k_out)
        self.linear_v = nn.Linear(3, v_out)
        # self.linear_energy = nn.Linear(3, num_points)
        # self.linear_qkv = nn.Linear(3, self.v_depth)

    def forward(self, coordinate, x, neighbors, Att_Score_method, pos_method):
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.global_split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N)
        k = self.global_split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N)
        v = self.global_split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, D)

        if Att_Score_method == 'dot_product':
            k = k.permute(0, 1, 3, 2)
            # k.shape == (B, H, D, N)
            if pos_method == 'method_i':
                energy = q @ k
                # energy.shape == (B, H, N, N)
                energy = self.pe_type_energy(coordinate, energy)
                v_context = self.pe_type_v_comb(coordinate, v)
                scale_factor = math.sqrt(q.shape[-1])
                attention = self.softmax(energy / scale_factor)
                # attention.shape == (B, H, N, N)
                x = (attention * v_context).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)
            elif pos_method == 'method_ii':
                q_context = self.pe_type_q_comb(coordinate, q)
                v_context = self.pe_type_v_comb(coordinate, v)
                energy = q @ k + q_context
                # energy.shape == (B, H, N, N)
                scale_factor = math.sqrt(q.shape[-1])
                attention = self.softmax(energy / scale_factor)
                # attention.shape == (B, H, N, N)
                x = (attention * v_context).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)
            elif pos_method == 'method_iii':
                q_context = self.pe_type_q_comb(coordinate, q)
                k_context = self.pe_type_k_comb(coordinate, k)
                v_context = self.pe_type_v_comb(coordinate, v)
                energy = q @ k + q_context + k_context
                # energy.shape == (B, H, N, N)
                scale_factor = math.sqrt(q.shape[-1])
                attention = self.softmax(energy / scale_factor)
                # attention.shape == (B, H, N, N)
                x = (attention * v_context).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)
            else:
                raise ValueError(f"pos_method must be 'method_i', 'method_ii' or 'method_iii'. Got: {pos_method}")

        elif Att_Score_method == 'subtraction':
            if pos_method == 'method_i':
                energy = (q - k) @ ((q - k).permute(1, 0))
                # energy.shape == (B, H, N, N)
                energy = self.pe_type_energy(coordinate, energy)
                v_context = self.pe_type_v_comb(coordinate, v)
                scale_factor = math.sqrt(q.shape[-1])
                attention = self.softmax(energy / scale_factor)
                # attention.shape == (B, H, N, N)
                x = (attention * v_context).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)
            elif pos_method == 'method_ii':
                q_context = self.pe_type_q_comb(coordinate, q)
                energy = (q - k) @ ((q - k).permute(1, 0)) + q_context
                # energy.shape == (B, H, N, N)
                v_context = self.pe_type_v_comb(coordinate, v)
                scale_factor = math.sqrt(q.shape[-1])
                attention = self.softmax(energy / scale_factor)
                # attention.shape == (B, H, N, N)
                x = (attention * v_context).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)
            elif pos_method == 'method_iii':
                q_context = self.pe_type_q_comb(coordinate, q)
                k_context = self.pe_type_k_comb(coordinate, k)
                v_context = self.pe_type_v_comb(coordinate, v)
                energy = (q - k) @ ((q - k).permute(1, 0)) + q_context + k_context
                # energy.shape == (B, H, N, N)
                scale_factor = math.sqrt(q.shape[-1])
                attention = self.softmax(energy / scale_factor)
                # attention.shape == (B, H, N, N)
                x = (attention * v_context).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)
            else:
                raise ValueError(f"pos_method must be 'method_i', 'method_ii' or 'method_iii'. Got: {pos_method}")

        elif Att_Score_method == 'addition':
            if pos_method == 'method_i':
                q = q.unsqueeze(-2).repeat(1, 1, 1, q.shape[2], 1)
                # q.shape == (B, H, N, N, D)
                k = k.unsqueeze(-2).repeat(1, 1, 1, k.shape[2], 1)
                # k.shape == (B, H, N, N, D)
                energy = q + k
                # energy.shape == (B, H, N, N, D)
                scale_factor = math.sqrt(q.shape[-1])
                energy = torch.tanh(energy / scale_factor)
                # attention.shape == (B, H, N, N, D)
                energy = energy @ self.p_add
                energy = self.pe_type_energy(coordinate, energy)
                attention = self.softmax(energy)
                # attention.shape == (B, H, N, N)
                x = (attention * v).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)
            elif pos_method == 'method_ii':
                q_context = self.pe_type_q_comb(coordinate, q)
                v_context = self.pe_type_v_comb(coordinate, v)
                # q_context.shape == (B, H, N, N)
                q = q.unsqueeze(-2).repeat(1, 1, 1, q.shape[2], 1)
                # q.shape == (B, H, N, N, D)
                k = k.unsqueeze(-2).repeat(1, 1, 1, k.shape[2], 1)
                # k.shape == (B, H, N, N, D)
                energy = q + k
                # energy.shape == (B, H, N, N, D)
                scale_factor = math.sqrt(q.shape[-1])
                energy = torch.tanh(energy / scale_factor)
                # attention.shape == (B, H, N, N, D)
                energy = energy @ self.p_add + q_context
                # energy.shape == (B, H, N, N)
                attention = self.softmax(energy)
                # attention.shape == (B, H, N, N)
                x = (attention * v_context).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)
            elif pos_method == 'method_iii':
                q_context = self.pe_type_q_comb(coordinate, q)
                k_context = self.pe_type_k_comb(coordinate, k)
                v_context = self.pe_type_v_comb(coordinate, v)
                # q_context.shape == (B, H, N, N)
                q = q.unsqueeze(-2).repeat(1, 1, 1, q.shape[2], 1)
                # q.shape == (B, H, N, N, D)
                k = k.unsqueeze(-2).repeat(1, 1, 1, k.shape[2], 1)
                # k.shape == (B, H, N, N, D)
                energy = q + k
                # energy.shape == (B, H, N, N, D)
                scale_factor = math.sqrt(q.shape[-1])
                energy = torch.tanh(energy / scale_factor)
                # attention.shape == (B, H, N, N, D)
                energy = energy @ self.p_add + q_context + k_context
                # energy.shape == (B, H, N, N)
                attention = self.softmax(energy)
                # attention.shape == (B, H, N, N)
                x = (attention * v_context).permute(0, 2, 1, 3)
                # x.shape == (B, N, H, D)

            else:
                raise ValueError(f"pos_method must be 'method_i', 'method_ii' or 'method_iii'. Got: {pos_method}")

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, N)
        return x

    def global_split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        x = x.permute(0, 1, 3, 2)
        # x.shape == (B, H, N, D)
        return x

    def pe_type_energy(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N)
        coordinates = coordinates.permute(0, 2, 1)
        # coordinates.shape == (B, N, 3)
        pos = self.linear_energy(coordinates)
        # pos.shape == (B, N, N x H), The last dimension is coordinate projection
        pos = pos.permute(0, 2, 1)
        # pos.shape == (B, N x H, N)
        pos = self.global_split_heads(pos, self.num_heads, self.num_points)
        # pos.shape == (B, H, N, N)
        pos_trans_value = trans_value + pos
        return pos_trans_value

    def pe_type_q_comb(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N)
        coordinates = coordinates.permute(0, 2, 1)
        # coordinates.shape == (B, N, 3)
        pos = self.linear_q(coordinates)
        # pos.shape == (B, N, D x H), The last dimension is coordinate projection
        pos = pos.permute(0, 2, 1)
        pos = self.global_split_heads(pos, self.num_heads, self.q_depth)
        # pos.shape == (B, H, N, D)
        pos_trans_value = trans_value @ pos.permute(0, 1, 3, 2)
        # pos_trans_value.shape == (B, H, N, N)
        return pos_trans_value

    def pe_type_k_comb(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N)
        coordinates = coordinates.permute(0, 2, 1)
        # coordinates.shape == (B, N, 3)
        pos = self.linear_k(coordinates)
        # pos.shape == (B, N, D x H), The last dimension is coordinate projection
        pos = pos.permute(0, 2, 1)
        pos = self.global_split_heads(pos, self.num_heads, self.k_depth)
        # pos.shape == (B, H, N, D)
        pos_trans_value = trans_value @ pos.permute(0, 1, 3, 2)
        # pos_trans_value.shape == (B, H, N, N)
        return pos_trans_value

    def pe_type_v_comb(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N)
        coordinates = coordinates.permute(0, 2, 1)
        # coordinates.shape == (B, N, 3)
        pos = self.linear_v(coordinates)
        # pos.shape == (B, N, D x H), The last dimension is coordinate projection
        pos = pos.permute(0, 2, 1)
        pos = self.global_split_heads(pos, self.num_heads, self.v_depth)
        # pos.shape == (B, H, N, D)
        pos_trans_value = trans_value + pos
        return pos_trans_value


class CrossAttention(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_points=1024, num_heads=8
                 , neighbor_type='diff', Att_Score_method='dot_product'):
        super(CrossAttention, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out != v_out:
            raise ValueError('Please check the dimension of energy')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        # print('q_out, k_out, v_out are same')
        self.Att_Score_method = Att_Score_method
        self.neighbor_type = neighbor_type
        self.num_points = num_points
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)
        self.softmax = nn.Softmax(dim=-1)

        if self.neighbor_type == 'diff' or self.neighbor_type == 'neighbor':
            self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
            self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
            self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        elif self.neighbor_type == 'center_neighbor' or self.neighbor_type == 'center_diff' or self.neighbor_type == 'neighbor_diff':
            self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
            self.k_conv = nn.Conv2d(2 * k_in, k_out, 1, bias=False)
            self.v_conv = nn.Conv2d(2 * v_in, v_out, 1, bias=False)
        elif self.neighbor_type == 'center_neighbor_diff':
            self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
            self.k_conv = nn.Conv2d(3 * k_in, k_out, 1, bias=False)
            self.v_conv = nn.Conv2d(3 * v_in, v_out, 1, bias=False)

        """
        self.p_add = nn.Parameter(torch.ones(1, self.k_depth), requires_grad=True)  # q.shape == (1, D)
        self.p_cat = nn.Parameter(torch.ones(1, 2 * self.k_depth))  # q.shape == (1, 2D)
        """

    def forward(self, pcd, neighbors):
        # pcd.shape == (B, C, N)
        pcd = pcd[:, :, :, None]  # pcd.shape == (B, C, N, 1)
        # neighbors.shape == (B, C, N, K)
        q = self.q_conv(pcd)
        # q.shape == (B, C, N, 1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, 1, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N, K)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, K, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N, K)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, K, D)
        k = k.permute(0, 1, 2, 4, 3)
        # k.shape == (B, H, N, D, K)

        if self.Att_Score_method == 'dot_product':
            energy = q @ k
            # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)
            # attention.shape == (B, H, N, 1, K)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
            # x.shape == (B, N, H, D)

        elif self.Att_Score_method == 'subtraction':
            q_repeated = q.repeat(1, 1, 1, k.shape[-2], 1)
            # q_repeated.shape == (B, H, N, K, D) to match with k
            energy = q_repeated - k
            # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)
            # attention.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)
            # x.shape == (B, N, H, K, D)
            x = x.sum(dim=-2)
            # x.shape == (B, N, H, D)
            """
        elif self.Att_Score_method == 'addition':
            q_repeated = q.repeat(1, 1, 1, None, 1)
            # q_repeated.shape == (B, H, N, 1, D)
            energy = q_repeated + k
            # q_repeated.shape == (B, H, N, K, D) to match with k
            # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = torch.tanh(energy / scale_factor)
            # attention.shape == (B, H, N, K, D)
            attention = attention @ self.p_add  # position encoding in here
            # attention.shape == (B, H, N, K)
            attention = self.softmax(attention.unsqueeze(-2))
            # attention.shape == (B, H, N, 1, K)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
            # x.shape == (B, N, H, D)

        elif self.Att_Score_method == 'concat':
            q_repeated = q.repeat(1, 1, 1, k.shape[-2], 1)
            # q_repeated.shape == (B, H, N, K, D) to match with k
            energy = torch.cat((q_repeated, k), dim=-1)
            # energy.shape == (B, H, N, K, 2D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = torch.tanh(energy / scale_factor)
            # attention.shape == (B, H, N, K, 2D)
            attention = attention @ self.p_cat  # position encoding in here
            # attention.shape == (B, H, N, K)
            attention = self.softmax(attention.unsqueeze(-2))
            # attention.shape == (B, H, N, 1, K)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
            # x.shape == (B, N, H, D)
            """
        else:
            raise ValueError(f'Invalid value for Att_Score_method: {self.Att_Score_method}')

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, N)
        # print("x.shape == (B, C, N)", x.shape)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x


"""
    def pe_sub_energy(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N, K)
        coordinates = coordinates.permute(0, 3, 2, 1)
        # coordinates.shape == (B, K, N, 3)
        pos = self.sub_linear_energy(coordinates)
        # pos.shape == (B, K, N, H x D)
        pos = pos.permute(0, 3, 2, 1)
        # pos.shape == (B, H x D, N, K)
        pos = self.split_heads(pos, self.num_heads, self.q_depth)
        # pos.shape == (B, H, N, K, D)
        pos_trans_value = trans_value + pos
        return pos_trans_value

    def pe_dot_energy(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N, K)
        coordinates = coordinates.permute(0, 3, 2, 1)
        # coordinates.shape == (B, K, N, 3)
        pos = self.dot_linear_energy(coordinates)
        # pos.shape == (B, K, N, H)
        pos = pos[:, :, :, None, :].permute(0, 4, 2, 3, 1)
        # pos.shape == (B, H, N, 1, K)
        pos_trans_value = trans_value + pos
        return pos_trans_value

    def pe_add_cat_energy(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N, K)
        coordinates = coordinates.permute(0, 3, 2, 1)
        # coordinates.shape == (B, K, N, 3)
        pos = self.dot_linear_energy(coordinates).permute(0, 3, 2, 1)
        # pos.shape == (B, H, N, K)
        pos_trans_value = trans_value + pos
        return pos_trans_value

    def pe_dot_q_comb(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N, K)
        coordinates = coordinates.permute(0, 3, 2, 1)
        # coordinates.shape == (B, K, N, 3)
        pos = self.linear_q(coordinates)
        # pos.shape == (B, K, N, D x H), The last dimension is coordinate projection
        pos = pos.permute(0, 3, 2, 1)
        pos = self.split_heads(pos, self.num_heads, self.q_depth)
        # pos.shape == (B, H, N, K, D)
        pos_trans_value = trans_value @ pos
        # pos_trans_value.shape == (B, H, N, 1, K)
        return pos_trans_value

    def pe_add_cat_q_comb(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N, K)
        coordinates = coordinates.permute(0, 3, 2, 1)
        # coordinates.shape == (B, K, N, 3)
        pos = self.linear_q(coordinates)
        # pos.shape == (B, K, N, D x H), The last dimension is coordinate projection
        pos = pos.permute(0, 3, 2, 1)
        pos = self.split_heads(pos, self.num_heads, self.q_depth)
        # pos.shape == (B, H, N, K, D)
        pos_trans_value = pos @ trans_value
        # pos_trans_value.shape == (B, H, N, K)
        return pos_trans_value

    "Miss three function for k for position encoding"

    def pe_v_comb(self, coordinates, trans_value):
        # coordinates.shape == (B, 3, N, K)
        coordinates = coordinates.permute(0, 3, 2, 1)
        # coordinates.shape == (B, K, N, 3)
        pos = self.linear_q(coordinates)
        # pos.shape == (B, K, N, D x H), The last dimension is coordinate projection
        pos = pos.permute(0, 3, 2, 1)
        pos = self.split_heads(pos, self.num_heads, self.q_depth)
        # pos.shape == (B, H, N, K, D)
        pos_trans_value = trans_value + pos
        # pos_trans_value.shape == (B, H, N, K, D)
        return pos_trans_value
    """


#  multi_scale as separate keys
class CrossAttentionMS(nn.Module):
    def __init__(self, key_one_or_sep, K, scale, neighbor_selection_method, neighbor_type, shared_ca, concat_ms_inputs,
                 mlp_or_ca,
                 q_in, q_out, k_in, k_out, v_in,
                 v_out, num_heads, ff_conv1_channels_in,
                 ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out):
        super(CrossAttentionMS, self).__init__()
        self.key_one_or_sep = key_one_or_sep
        self.K = K
        self.scale = scale
        self.neighbor_selection_method = neighbor_selection_method
        self.neighbor_type = neighbor_type
        self.shared_ca = shared_ca
        self.concat_ms_inputs = concat_ms_inputs
        self.mlp_or_ca = mlp_or_ca

        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')
        if concat_ms_inputs:
            if not shared_ca:
                raise ValueError('shared_ca must be true when concat_ms_inputs is true')
        if shared_ca:
            self.ca = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads)
        else:
            self.ca_list = nn.ModuleList(
                [CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads) for _ in range(scale + 1)])
        if not concat_ms_inputs:
            if mlp_or_ca == 'mlp':
                self.linear = nn.Conv1d(v_out * (scale + 1), q_in, 1, bias=False)
            elif mlp_or_ca == 'ca':
                self.ca_aggregation = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads)
            else:
                raise ValueError(f'mlp_or_ca should be mlp or ca, but got {mlp_or_ca}')
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, pcd, coordinate):
        if self.key_one_or_sep == 'one':
            neighbors, idx_all = ops.select_neighbors_in_one_key(pcd, coordinate, self.K, self.scale,
                                                                 self.neighbor_selection_method, self.neighbor_type)
            neighbors = neighbors.permute(0, 3, 1, 2)  # neighbor.shape == (B, num x C, N, (scale+1) x K)
        elif self.key_one_or_sep == 'sep':
            neighbors, idx_all = ops.select_neighbors_in_separate_key(pcd, coordinate, self.K, self.scale,
                                                                      self.neighbor_selection_method,
                                                                      self.neighbor_type)
            neighbors = neighbors.permute(0, 3, 1, 2)  # neighbor.shape == (B, num x C, N, (scale+1) x K)
            neighbor_list = ops.list_generator(neighbors, self.K, self.scale)
            # element shape in list == (B, num x C, N, K)
            # print("neighbor_list", neighbor_list.shape)
        if self.concat_ms_inputs:
            neighbors = torch.cat([neighbors], dim=1)
            x_out = self.ca(pcd, neighbors)
        else:
            x_output_list = []
            if self.shared_ca:
                for neighbors in neighbor_list:
                    # x.shape == (B, C, N)  neighbors.shape == (B, C, N, K)
                    x_out = self.ca(pcd, neighbors)
                    # x_out.shape == (B, C, N)
                    x_output_list.append(x_out)
            else:
                for neighbors, ca in zip(neighbor_list, self.ca_list):
                    # x.shape == (B, C, N)  neighbors.shape == (B, C, N, K)
                    print("neighbors", neighbors.shape)
                    x_out = ca(pcd, neighbors)
                    # x_out.shape == (B, C, N)
                    x_output_list.append(x_out)
            if self.mlp_or_ca == 'mlp':
                x_out = torch.cat(x_output_list, dim=1)
                # x_out.shape == (B, C, N)
                x_out = self.linear(x_out)
                # x_out.shape == (B, C, N)
            else:
                neighbors = torch.stack(x_output_list, dim=-1)
                # x.shape == (B, C, N)   neighbors.shape == (B, C, N, scale+1)
                x_out = self.ca_aggregation(pcd, neighbors)
                # x_out.shape == (B, C, N)
        # x_out.shape == (B, C, N)
        # print("pcd", pcd.shape, "x_out", x_out.shape)
        x = self.bn1(pcd + x_out)
        # x.shape == (B, C, N)
        x_out = self.ff(x)
        # x_out.shape == (B, C, N)
        x = self.bn2(x + x_out)
        # x.shape == (B, C, N)
        return x


# multi_scale as one key
class CrossAttentionMS_OneK(nn.Module):
    def __init__(self, q_in, q_out, k_in, k_out, v_in, v_out, scale, num_heads, group_type):
        super(CrossAttentionMS_OneK, self).__init__()

        self.head_dim = q_in // num_heads
        self.num_heads = num_heads
        self.query_transform = nn.Linear(q_in, q_out)  # Linear layer (Wq)
        self.key_transforms = nn.ModuleList(
            [nn.Linear(k_in, k_out) for _ in range(scale + 1)])  # Each scale has an independent
        # linear layer for key transformation (Wk)
        self.value_transform = nn.Linear(v_in, v_out)  # Linear layer (Wv)
        self.scale = scale
        self.group_type = group_type

    def forward(self, pcd, coordinate, K, neighbor_selection_method, neighbor_type, operation_mode='multiple'):
        # Input_point_cloud_shape: (B, N, C),
        # multi_scale_neighbors is list, len == num_scales, each element shape: (B, C, N, scale_i)

        agg_neighbors = ops.select_neighbors_in_one_key(pcd, coordinate, K, self.scale, neighbor_selection_method
                                                        , self.group_type)
        # split the input tensor into multiple heads
        query = self.query_transform(pcd).view(pcd.shape[0], pcd.shape[1], self.num_heads, self.head_dim)  # query
        # shape:(B, N, num_heads, head_dim)

        keys = [
            self.key_transforms[i](agg_neighbors[i]).view(*agg_neighbors[i].shape[:-1], self.num_heads,
                                                          self.head_dim) for i in range(self.scale + 1)]
        # keys is list, len == num_scales, each element shape: (B, num_heads, N, scale_i, head_dim)

        attention_maps = ops.operation_mode(operation_mode, query, keys)

        # sum of attention_map
        attention_map = sum(attention_maps)  # attention_map shape: (B, N, num_heads, max_scale)

        value = self.value_transform(pcd).view(pcd.shape[0], pcd.shape[1], self.num_heads, self.head_dim).unsqueeze(
            -2)  # value shape: (B, N, num_heads, 1, head_dim)

        output = torch.matmul(attention_map.unsqueeze(-2), value).squeeze(-2)  # output shape: (B, N, num_heads,
        # head_dim)

        # concatenate the heads
        output = output.reshape(output.shape[0], output.shape[1], -1)

        return output  # (B, N, num_heads * head_dim)


class Point_Embedding(nn.Module):
    def __init__(self, embedding_k, conv1_channel_in, conv1_channel_out, conv2_channel_in, conv2_channel_out):
        super(Point_Embedding, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(conv1_channel_in, conv1_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv1_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv2_channel_in, conv2_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv2_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.embedding_k = embedding_k

    # init_a.shape == (B, 3, N)
    # init_b.shape == (B, 3, N)
    def forward(self, a, b):
        # print("a initial shape:", a.shape)
        # print("b initial shape:", b.shape)
        a = a.permute(0, 2, 1)  # a.shape == (B, N, 3)
        b = b.permute(0, 2, 1)  # b.shape == (B, N, 3)
        # print("a after permute shape:", a.shape)
        # print("b after permute shape:", b.shape)
        idx = ops.knn(a, b, self.embedding_k)  # idx.shape == (B, N, K)
        # print("idx shape:", idx.shape)
        neighbors = ops.index_points(a, idx)  # neighbors.shape == (B, N, K, 3)
        # print("neighbors shape:", neighbors.shape)
        diff = neighbors - a[:, :, None, :]  # diff.shape == (B, N, K, 3)
        # print("diff shape:", diff.shape)
        a_exp = a.unsqueeze(2).expand(-1, -1, self.embedding_k, -1)  # a_exp.shape == (B, N, K, 3)
        # print("a_exp shape:", a_exp.shape)
        x = torch.cat([diff, a_exp], dim=3).permute(0, 3, 1, 2)  # x.shape == (B, 6, N, K)
        # print("x after cat shape:", x.shape)
        x = self.conv1(x)  # x.shape == (B, C, N, K)
        x = self.conv2(x)  # x.shape == (B, C, N, K)
        # print("x after conv1 shape:", x.shape)
        x, _ = x.max(dim=3)  # x.shape == (B, C, N)
        # print("x after max shape:", x.shape)
        return x


class ModelNetModel(nn.Module):
    def __init__(self, embedding_k, point_emb1_in, point_emb1_out, conv2_channel_in, conv2_channel_out, key_one_or_sep,
                 K, scale, neighbor_selection_method, neighbor_type, shared_ca,
                 concat_ms_inputs, mlp_or_ca, q_in, q_out, k_in, k_out,
                 v_in, v_out, num_heads, ff_conv1_channels_in,
                 ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out):
        super(ModelNetModel, self).__init__()
        self.k_out = k_out[0]
        self.num_att_layer = len(k_out)
        self.Point_Embedding_list = nn.ModuleList(
            [Point_Embedding(embedding_k, point_emb1_in, point_emb1_out, conv2_channel_in, conv2_channel_out) for
             embedding_k, point_emb1_in, point_emb1_out, conv2_channel_in, conv2_channel_out in
             zip(embedding_k, point_emb1_in, point_emb1_out, conv2_channel_in, conv2_channel_out)])
        self.CrossAttentionMS_list = nn.ModuleList(
            [CrossAttentionMS(key_one_or_sep, K, scale, neighbor_selection_method, neighbor_type, shared_ca,
                              concat_ms_inputs, mlp_or_ca, q_in, q_out, k_in, k_out,
                              v_in, v_out, num_heads, ff_conv1_channels_in,
                              ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out) for
             key_one_or_sep, K, scale, neighbor_selection_method, neighbor_type, shared_ca,
             concat_ms_inputs, mlp_or_ca, q_in, q_out, k_in, k_out,
             v_in, v_out, num_heads, ff_conv1_channels_in,
             ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out in
             zip(key_one_or_sep, K, scale, neighbor_selection_method, neighbor_type, shared_ca,
                 concat_ms_inputs, mlp_or_ca, q_in, q_out, k_in, k_out,
                 v_in, v_out, num_heads, ff_conv1_channels_in,
                 ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)])

        self.linear1 = nn.Sequential(nn.Linear(self.num_att_layer * 1024, 1024), nn.BatchNorm1d(1024),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))
        self.linear3 = nn.Linear(256, 40)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.conv_list = nn.ModuleList([nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False) for channel_in in ff_conv2_channels_out])

    def forward(self, x):
        device = x.device
        xyz = x
        # print("my input :", x.shape)
        """
        :original pc xyz: xyz.shape == (B, 3, N)
        """
        x_list = []
        res_link_list = []
        for point_embedding in self.Point_Embedding_list:
            x = point_embedding(x, x)  # x.shape == (B, C, N)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)  # x.shape == (B, num_layer x C, N)

        i = 0
        for cross_attentionMS in self.CrossAttentionMS_list:
            x = cross_attentionMS(x, xyz)
            # x.shape == (B, C, N)
            # x_extract = x.max(dim=-1)[0]
            x_expand = self.conv_list[i](x).max(dim=-1)[0]  # x_expand.shape == (B, 1024)
            res_link_list.append(x_expand)
            # print("x.shape", x.shape)
            # i += 1
        x = torch.cat(res_link_list, dim=1)  # x.shape == (B, 4096)
        x = self.linear1(x)
        # x.shape == (B, 1024)
        x = self.dp1(x)
        # x.shape == (B, 1024)
        x = self.linear2(x)
        # x.shape == (B, 256)
        x = self.dp2(x)
        # x.shape == (B, 256)
        x = self.linear3(x)
        # x.shape == (B, 40)
        return x
