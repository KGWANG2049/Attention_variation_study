import torch
from torch import nn
import math

"""
Local based attention variation input shape：
q.shape == (B, H, N, 1, D)
k.shape == (B, H, N, D, K)
v.shape == (B, H, N, K, D)
"""


class Pe_local_based_attention_variation(nn.Module):
    def __init__(self, pe_method, k_out=64, num_heads=8, att_score_method='local_scalar_dot'):
        super(Pe_local_based_attention_variation, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.k_depth = int(k_out / num_heads)
        self.pe_method = pe_method
        if att_score_method == 'local_scalar_dot' or att_score_method == 'local_scalar_sub':
            if pe_method == 'pe_ii':
                self.linear1 = nn.Linear(3, num_heads)
                self.linear2 = nn.Linear(3, k_out)
            elif pe_method == 'pe_iii':
                self.linear1 = nn.Linear(3, k_out)
                self.linear2 = nn.Linear(3, k_out)
            elif pe_method == 'pe_iv':
                self.linear1 = nn.Linear(3, k_out)
                self.linear2 = nn.Linear(3, k_out)
                self.linear3 = nn.Linear(3, k_out)

        elif att_score_method == 'local_scalar_add':
            if pe_method == 'pe_ii':
                self.linear1 = nn.Linear(3, num_heads)
                self.linear2 = nn.Linear(3, k_out)
            elif pe_method == 'pe_iii':
                self.linear1 = nn.Linear(3, k_out)
                self.linear2 = nn.Linear(3, k_out)
            elif pe_method == 'pe_iv':
                self.linear1 = nn.Linear(3, k_out)
                self.linear2 = nn.Linear(3, k_out)
                self.linear3 = nn.Linear(3, k_out)

            self.p_add = nn.Parameter(torch.ones(self.k_depth, 1))  # p_add.shape == (D, 1)
        elif att_score_method == 'local_scalar_cat':
            if pe_method == 'pe_ii':
                self.linear1 = nn.Linear(3, num_heads)
                self.linear2 = nn.Linear(3, k_out)
            elif pe_method == 'pe_iii':
                self.linear1 = nn.Linear(3, k_out)
                self.linear2 = nn.Linear(3, k_out)
            elif pe_method == 'pe_iv':
                self.linear1 = nn.Linear(3, k_out)
                self.linear2 = nn.Linear(3, k_out)
                self.linear3 = nn.Linear(3, k_out)

            self.p_cat = nn.Parameter(torch.ones(2 * self.k_depth, 1))  # p_cat.shape == (1, 2D)
        elif att_score_method == 'local_vector_sub' or att_score_method == 'local_vector_add':
            if pe_method == 'pe_ii':
                self.linear1 = nn.Linear(3, k_out)
                self.linear2 = nn.Linear(3, k_out)
            elif pe_method == 'pe_iii':
                self.linear1 = nn.Linear(3, num_heads)
                self.linear2 = nn.Linear(3, k_out)
            elif pe_method == 'pe_iv':
                self.linear1 = nn.Linear(3, num_heads)
                self.linear2 = nn.Linear(3, k_out)

    def local_attention_scalarDot(self, q, k, v, re_xyz, xyz=None):  # re_xyz.shape == (B, N, K, 3), xyz.shape == (B, 3, N)
        if self.pe_method == 'pe_ii':
            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, H, N, K)
            q_xyz = q_xyz.unsqueeze(3)  # q_xyz.shape == (B, H, N, 1, K)
            energy = q @ k + q_xyz  # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(
                energy / scale_factor)  # attention.shape == (B, H, N, 1, K), v.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2],
                               self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        elif self.pe_method == 'pe_iii':
            q_xyz = self.linear1(re_xyz)  # q_xyz.shape == (B, N, K, C), q.shape == (B, H, N, 1, D)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, N, D, K)
            q_xyz = q @ q_xyz  # energy.shape == (B, H, N, 1, K)
            energy = q @ k + q_xyz  # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(
                energy / scale_factor)  # attention.shape == (B, H, N, 1, K), v.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2],
                               self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iv':  # k.shape == (B, H, N, D, K) # xyz.shape == (B, 3, N)
            q_xyz = self.linear1(re_xyz)  # q_xyz.shape == (B, N, K, C), q.shape == (B, H, N, 1, D)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth, q_xyz.shape[2])  # q_xyz.shape == (B, H, N, D, K)
            q_xyz = q @ q_xyz  # energy.shape == (B, H, N, 1, K)
            k_xyz = self.linear3(xyz.permute(0, 2, 1))  # k_xyz.shape == (B, N, C)
            k_xyz = k_xyz.view(k_xyz.shape[0], self.num_heads, k_xyz.shape[1], 1,
                               self.k_depth)  # k_xyz.shape == (B, H, N, 1, D)
            k_xyz = k_xyz @ k  # k_xyz.shape == (B, H, N, 1, K)
            energy = q @ k + q_xyz + k_xyz  # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K), v.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2],
                               self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        return x

    def local_attention_scalarSub(self, q, k, v, re_xyz, xyz=None):
        if self.pe_method == 'pe_ii':
            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, N, K, C=H)
            q_xyz = q_xyz.unsqueeze(3)  # q_xyz.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1],
                                  1)  # q_repeated.shape == (B, H, N, K, D) k.shape == (B, H, N, D, K)
            diff = q_repeated - k.permute(0, 1, 2, 4, 3)  # diff.shape == (B, H, N, K, D)
            energy = q @ diff.permute(0, 1, 2, 4, 3) + q_xyz  # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, 1, D)
            x = x[:, :, :, 0, :]  # x.shape == (B, N, H, D)
        elif self.pe_method == 'pe_iii':
            q_xyz = self.linear1(re_xyz)  # q_xyz.shape == (B, N, K, C), q.shape == (B, H, N, 1, D)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, N, D, K)
            q_xyz = q @ q_xyz  # energy.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1],
                                  1)  # q_repeated.shape == (B, H, N, K, D) k.shape == (B, H, N, D, K)
            diff = q_repeated - k.permute(0, 1, 2, 4, 3)  # diff.shape == (B, H, N, K, D)
            energy = q @ diff.permute(0, 1, 2, 4, 3) + q_xyz  # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, 1, D)
            x = x[:, :, :, 0, :]  # x.shape == (B, N, H, D)
        elif self.pe_method == 'pe_iv':
            q_xyz = self.linear1(re_xyz)  # q_xyz.shape == (B, N, K, C), q.shape == (B, H, N, 1, D)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, N, D, K)
            q_xyz = q @ q_xyz  # energy.shape == (B, H, N, 1, K)
            k_xyz = self.linear3(xyz.permute(0, 2, 1))  # k_xyz.shape == (B, N, C)
            k_xyz = k_xyz.view(k_xyz.shape[0], self.num_heads, k_xyz.shape[1], 1,
                               self.k_depth)  # k_xyz.shape == (B, H, N, 1, D)
            k_xyz = k_xyz @ k  # k_xyz.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1],
                                  1)  # q_repeated.shape == (B, H, N, K, D) k.shape == (B, H, N, D, K)
            diff = q_repeated - k.permute(0, 1, 2, 4, 3)  # diff.shape == (B, H, N, K, D)
            energy = q @ diff.permute(0, 1, 2, 4, 3) + q_xyz + k_xyz  # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, 1, D)
            x = x[:, :, :, 0, :]  # x.shape == (B, N, H, D)
        return x

    def local_attention_scalarAdd(self, q, k, v, re_xyz, xyz=None):
        if self.pe_method == 'pe_ii':
            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, N, K, C=H)
            q_xyz = q_xyz.unsqueeze(3)  # q_xyz.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = q_repeated + k.permute(0, 1, 2, 4, 3)  # energy.shape == (B, H, N, K, D)
            energy = torch.tanh(energy)  # attention.shape == (B, H, N, K, D)
            energy = energy @ self.p_add  # attention.shape == (B, H, N, K, 1)
            energy = energy.permute(0, 1, 2, 4, 3) + q_xyz  # attention.shape == (B, H, N, 1, K)
            attention = self.softmax(energy)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        elif self.pe_method == 'pe_iii':
            q_xyz = self.linear1(re_xyz)  # q_xyz.shape == (B, N, K, C), q.shape == (B, H, N, 1, D)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, N, D, K)
            q_xyz = q @ q_xyz  # energy.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = q_repeated + k.permute(0, 1, 2, 4, 3)  # energy.shape == (B, H, N, K, D)
            energy = torch.tanh(energy)  # attention.shape == (B, H, N, K, D)
            energy = energy @ self.p_add  # attention.shape == (B, H, N, K, 1)
            energy = energy.permute(0, 1, 2, 4, 3) + q_xyz  # attention.shape == (B, H, N, 1, K)
            attention = self.softmax(energy)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        elif self.pe_method == 'pe_iv':
            q_xyz = self.linear1(re_xyz)  # q_xyz.shape == (B, N, K, C), q.shape == (B, H, N, 1, D)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, N, D, K)
            q_xyz = q @ q_xyz  # energy.shape == (B, H, N, 1, K)
            k_xyz = self.linear3(xyz.permute(0, 2, 1))  # k_xyz.shape == (B, N, C)
            k_xyz = k_xyz.view(k_xyz.shape[0], self.num_heads, k_xyz.shape[1], 1,
                               self.k_depth)  # k_xyz.shape == (B, H, N, 1, D)
            k_xyz = k_xyz @ k  # k_xyz.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = q_repeated + k.permute(0, 1, 2, 4, 3)  # energy.shape == (B, H, N, K, D)
            energy = torch.tanh(energy)  # attention.shape == (B, H, N, K, D)
            energy = energy @ self.p_add  # attention.shape == (B, H, N, K, 1)
            energy = energy.permute(0, 1, 2, 4, 3) + q_xyz + k_xyz  # attention.shape == (B, H, N, 1, K)
            attention = self.softmax(energy)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        return x

    def local_attention_scalarCat(self, q, k, v, re_xyz, xyz=None):

        if self.pe_method == 'pe_ii':
            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, N, K, C=H)
            q_xyz = q_xyz.unsqueeze(3)  # q_xyz.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = torch.cat((q_repeated, k.permute(0, 1, 2, 4, 3)), dim=-1)  # energy.shape == (B, H, N, K, 2D)
            energy = torch.tanh(energy)  # attention.shape == (B, H, N, K, 2D)
            energy = energy @ self.p_cat  # attention.shape == (B, H, N, K, 1)
            energy = energy.permute(0, 1, 2, 4, 3) + q_xyz
            attention = self.softmax(energy)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iii':
            q_xyz = self.linear1(re_xyz)  # q_xyz.shape == (B, N, K, C), q.shape == (B, H, N, 1, D)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, N, D, K)
            q_xyz = q @ q_xyz  # energy.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = torch.cat((q_repeated, k.permute(0, 1, 2, 4, 3)), dim=-1)  # energy.shape == (B, H, N, K, 2D)
            energy = torch.tanh(energy)  # energy.shape == (B, H, N, K, 2D)
            energy = energy @ self.p_cat  # energy.shape == (B, H, N, K, 1)
            energy = energy.permute(0, 1, 2, 4, 3) + q_xyz
            attention = self.softmax(energy)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iv':
            q_xyz = self.linear1(re_xyz)  # q_xyz.shape == (B, N, K, C), q.shape == (B, H, N, 1, D)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, N, D, K)
            q_xyz = q @ q_xyz  # energy.shape == (B, H, N, 1, K)
            k_xyz = self.linear3(xyz.permute(0, 2, 1))  # k_xyz.shape == (B, N, C)
            k_xyz = k_xyz.view(k_xyz.shape[0], self.num_heads, k_xyz.shape[1], 1,
                               self.k_depth)  # k_xyz.shape == (B, H, N, 1, D)
            k_xyz = k_xyz @ k  # k_xyz.shape == (B, H, N, 1, K)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = torch.cat((q_repeated, k.permute(0, 1, 2, 4, 3)), dim=-1)  # energy.shape == (B, H, N, K, 2D)
            energy = torch.tanh(energy)  # energy.shape == (B, H, N, K, 2D)
            energy = energy @ self.p_cat  # energy.shape == (B, H, N, K, 1)
            energy = energy.permute(0, 1, 2, 4, 3) + q_xyz + k_xyz
            attention = self.softmax(energy)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        return x

    def local_attention_vectorSub(self, q, k, v, re_xyz):

        if self.pe_method == 'pe_ii':
            q_xyz = self.linear1(re_xyz)  # v_xyz.shape == (B, N, K, C)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], q_xyz.shape[2], self.k_depth)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = q_repeated - k.permute(0, 1, 2, 4, 3) + q_xyz  # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iii':
            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, H, N, K)
            q_xyz = q_xyz.unsqueeze(4)  # q_xyz.shape == (B, H, N, K, 1)
            q_xyz = q_xyz @ q  # energy.shape == (B, H, N, K, D)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = q_repeated - k.permute(0, 1, 2, 4, 3) + q_xyz  # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)  # x.shape == (B, N, H, D)
            print('im here')

        elif self.pe_method == 'pe_iv':
            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, H, N, K)
            q_xyz = q_xyz.unsqueeze(4)  # q_xyz.shape == (B, H, N, K, 1)
            q_xyz = q_xyz @ q  # energy.shape == (B, H, N, K, D)
            '''
            k_xyz = self.linear3(re_xyz)  # k_xyz.shape == (B, N, K, C)
            k_xyz = k_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth, q_xyz.shape[2]) # k_xyz.shape == (B, H ,N, D, K)
            k_xyz = k_xyz + k
            '''
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)
            # q_repeated.shape == (B, H, N, K, D) to match with k
            energy = q_repeated - k.permute(0, 1, 2, 4, 3) + q_xyz  # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)
            # attention.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)  # x.shape == (B, N, H, D)
        return x

    def local_attention_vectorAdd(self, q, k, v, re_xyz):

        if self.pe_method == 'pe_ii':
            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, H, N, K)
            q_xyz = q_xyz.unsqueeze(4)  # q_xyz.shape == (B, H, N, K, 1)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, 1, D)
            energy = q_repeated + k.permute(0, 1, 2, 4, 3) + q_xyz  # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iii':
            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, H, N, K)
            q_xyz = q_xyz.unsqueeze(4)  # q_xyz.shape == (B, H, N, K, 1)
            q_xyz = q_xyz @ q  # energy.shape == (B, H, N, K, D)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, 1, D)
            energy = q_repeated + k.permute(0, 1, 2, 4, 3) + q_xyz  # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)  # x.shape == (B, N, H, D)
        elif self.pe_method == 'pe_iv':

            q_xyz = self.linear1(re_xyz).permute(0, 3, 1, 2)  # q_xyz.shape == (B, H, N, K)
            q_xyz = q_xyz.unsqueeze(4)  # q_xyz.shape == (B, H, N, K, 1)
            q_xyz = q_xyz @ q  # energy.shape == (B, H, N, K, D)
            '''
            k_xyz = self.linear3(re_xyz)  # k_xyz.shape == (B, N, K, C)
            k_xyz = k_xyz.view(q_xyz.shape[0], self.num_heads, q_xyz.shape[1], self.k_depth, q_xyz.shape[2]) # k_xyz.shape == (B, H ,N, D, K)
            k_xyz = k_xyz + k
            
            '''
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, 1, D)
            energy = q_repeated + k.permute(0, 1, 2, 4, 3) + q_xyz  # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, K, D)
            v_xyz = self.linear2(re_xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)  # v_xyz.shape == (B, H, N, K, D)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)  # x.shape == (B, N, H, D)
        return x


"""
Global based attention variation input shape：
q.shape == (B, H, N, D)
k.shape == (B, H, N, D)
v.shape == (B, H, N, D)
"""


class Pe_global_based_attention_variation(nn.Module):
    def __init__(self, pe_method, num_heads, q_out):
        super(Pe_global_based_attention_variation, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.pe_method = pe_method
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        if pe_method == 'pe_ii':
            self.conv1 = nn.Conv1d(3, num_heads, 1, bias=False)
            self.conv2 = nn.Conv1d(3, q_out, 1, bias=False)
        elif pe_method == 'pe_iii':
            self.conv1 = nn.Conv1d(3, q_out, 1, bias=False)
            self.conv2 = nn.Conv1d(3, q_out, 1, bias=False)
        elif pe_method == 'pe_iv':
            self.conv1 = nn.Conv1d(3, q_out, 1, bias=False)
            self.conv2 = nn.Conv1d(3, q_out, 1, bias=False)
            self.conv3 = nn.Conv1d(3, q_out, 1, bias=False)

    def global_attention_Dot(self, q, k, v, xyz):
        if self.pe_method == 'pe_ii':
            q_xyz = self.conv1(xyz).unsqueeze(-1)  # x.shape == (B, H, N，1)
            q_xyz = q_xyz.repeat(1, 1, 1, q_xyz.shape[2])  # x.shape == (B, H, N, N)
            energy = q @ k.permute(0, 1, 3, 2) + q_xyz  # energy.shape == (B, H, N, N)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, N)
            v_xyz = self.conv2(xyz)  # v_xyz.shape == (B, C, N)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, self.q_depth, v_xyz.shape[2])  # v_xyz.shape == (B, H, D, N)
            v_xyz = v_xyz.permute(0, 1, 3, 2)  # v_xyz.shape == (B, H, N, D)
            v = v + v_xyz
            x = (attention @ v).permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iii':
            q_xyz = self.conv1(xyz)  # x.shape == (B, C, N)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, self.q_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, D, N)
            q_xyz = q @ q_xyz  # q_xyz.shape == (B, H, N, N)
            energy = q @ k.permute(0, 1, 3, 2) + q_xyz  # energy.shape == (B, H, N, N)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, N)
            v_xyz = self.conv2(xyz)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, self.q_depth, v_xyz.shape[2])  # v_xyz.shape == (B, H, D, N)
            v_xyz = v_xyz.permute(0, 1, 3, 2)  # v_xyz.shape == (B, H, N, D)
            v = v + v_xyz
            x = (attention @ v).permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iv':
            q_xyz = self.conv1(xyz)  # x.shape == (B, C, N)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, self.q_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, D, N)
            q_xyz = q @ q_xyz  # q_xyz.shape == (B, H, N, N)
            k_xyz = self.conv3(xyz)  # x.shape == (B, C, N)
            k_xyz = k_xyz.view(k_xyz.shape[0], self.num_heads, self.q_depth,
                               k_xyz.shape[2])  # q_xyz.shape == (B, H, D, N)
            k_xyz = k @ k_xyz  # k_xyz.shape == (B, H, N, N)
            energy = q @ k.permute(0, 1, 3, 2) + q_xyz + k_xyz  # energy.shape == (B, H, N, N)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, N)
            v_xyz = self.conv2(xyz)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, self.q_depth, v_xyz.shape[2])  # x.shape == (B, H, D, N)
            v_xyz = v_xyz.permute(0, 1, 3, 2)  # v_xyz.shape == (B, H, N, D)
            v = v + v_xyz
            x = (attention @ v).permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        return x

    def global_attention_Sub(self, q, k, v, xyz=None):
        if self.pe_method == 'pe_ii':
            q_xyz = self.conv1(xyz).unsqueeze(-1)  # x.shape == (B, H, N，1)
            q_xyz = q_xyz.repeat(1, 1, 1, q_xyz.shape[2])  # x.shape == (B, H, N, N)
            inner = -2 * torch.matmul(q, k.transpose(3, 2))  # inner.shape == (B, H, N, N)
            qq = torch.sum(q ** 2, dim=-1, keepdim=True)  # qq.shape == (B, H, N, 1)
            kk = torch.sum(k ** 2, dim=-1, keepdim=True)  # kk.shape == (B, H, N, 1)
            diff_squared = qq + inner + kk.transpose(3, 2)  # pairwise_distance.shape == (B, H, N, N)
            diff_squared = diff_squared + q_xyz  # diff_squared.shape == (B, H, N, N)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(-diff_squared / scale_factor)  # attention.shape == (B, H, N, N)
            v_xyz = self.conv2(xyz)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, self.q_depth, v_xyz.shape[2])  # v_xyz.shape == (B, H, D, N)
            v_xyz = v_xyz.permute(0, 1, 3, 2)  # v_xyz.shape == (B, H, N, D)
            v = v + v_xyz
            x = (attention @ v).permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iii':
            q_xyz = self.conv1(xyz)  # x.shape == (B, C, N)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, self.q_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, D, N)
            q_xyz = q @ q_xyz  # q_xyz.shape == (B, H, N, N)
            inner = -2 * torch.matmul(q, k.transpose(3, 2))  # inner.shape == (B, H, N, N)
            qq = torch.sum(q ** 2, dim=-1, keepdim=True)  # qq.shape == (B, H, N, 1)
            kk = torch.sum(k ** 2, dim=-1, keepdim=True)  # kk.shape == (B, H, N, 1)
            diff_squared = qq + inner + kk.transpose(3, 2)  # pairwise_distance.shape == (B, H, N, N)
            diff_squared = diff_squared + q_xyz  # diff_squared.shape == (B, H, N, N)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(-diff_squared / scale_factor)  # attention.shape == (B, H, N, N)
            v_xyz = self.conv2(xyz)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, self.q_depth, v_xyz.shape[2])  # v_xyz.shape == (B, H, D, N)
            v_xyz = v_xyz.permute(0, 1, 3, 2)  # v_xyz.shape == (B, H, N, D)
            v = v + v_xyz
            x = (attention @ v).permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)

        elif self.pe_method == 'pe_iv':
            q_xyz = self.conv1(xyz)  # x.shape == (B, C, N)
            q_xyz = q_xyz.view(q_xyz.shape[0], self.num_heads, self.q_depth,
                               q_xyz.shape[2])  # q_xyz.shape == (B, H, D, N)
            q_xyz = q @ q_xyz  # q_xyz.shape == (B, H, N, N)
            k_xyz = self.conv3(xyz)  # x.shape == (B, C, N)
            k_xyz = k_xyz.view(k_xyz.shape[0], self.num_heads, self.q_depth,
                               k_xyz.shape[2])  # q_xyz.shape == (B, H, D, N)
            k_xyz = k @ k_xyz  # k_xyz.shape == (B, H, N, N)
            inner = -2 * torch.matmul(q, k.transpose(3, 2))  # inner.shape == (B, H, N, N)
            qq = torch.sum(q ** 2, dim=-1, keepdim=True)  # qq.shape == (B, H, N, 1)
            kk = torch.sum(k ** 2, dim=-1, keepdim=True)  # kk.shape == (B, H, N, 1)
            diff_squared = qq + inner + kk.transpose(3, 2)  # pairwise_distance.shape == (B, H, N, N)
            diff_squared = diff_squared + q_xyz + k_xyz  # diff_squared.shape == (B, H, N, N)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(-diff_squared / scale_factor)  # attention.shape == (B, H, N, N)
            v_xyz = self.conv2(xyz)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, self.q_depth, v_xyz.shape[2])  # v_xyz.shape == (B, H, D, N)
            v_xyz = v_xyz.permute(0, 1, 3, 2)  # v_xyz.shape == (B, H, N, D)
            v = v + v_xyz
            x = (attention @ v).permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        return x
