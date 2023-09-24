import math
import torch
from torch import nn
from utils import ops
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
            self.linear1 = nn.Linear(3, num_heads)
            self.linear2 = nn.Linear(3, k_out)
        if att_score_method == 'local_scalar_add':
            self.linear1 = nn.Linear(3, num_heads)
            self.linear2 = nn.Linear(3, k_out)
            self.k_depth = int(k_out / num_heads)
            self.p_add = nn.Parameter(torch.ones(self.k_depth, 1))  # p_add.shape == (D, 1)
        elif att_score_method == 'local_scalar_cat':
            self.linear1 = nn.Linear(3, num_heads)
            self.linear2 = nn.Linear(3, k_out)
            self.k_depth = int(k_out / num_heads)
            self.p_cat = nn.Parameter(torch.ones(2 * self.k_depth, 1))  # p_cat.shape == (1, 2D)
        elif att_score_method == 'local_vector_sub':
            self.linear1 = nn.Linear(3, k_out)
            self.linear2 = nn.Linear(3, k_out)
        elif att_score_method == 'local_vector_add':
            self.linear1 = nn.Linear(3, k_out)
            self.linear2 = nn.Linear(3, k_out)

    def local_attention_scalarDot(self, q, k, v, xyz):  # xyz.shape == (B, N, K, 3) !
        if self.pe_method == 'pe_ii':
            energy_xyz = self.linear1(xyz).permute(0, 3, 1, 2)  # energy_xyz.shape == (B, N, K, C=H)
            energy_xyz = energy_xyz.unsqueeze(3)  # energy_xyz.shape == (B, H, N, 1, K)
            # print('energy_xyz', energy_xyz.shape)
            energy = q @ k + energy_xyz  # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)
            # attention.shape == (B, H, N, 1, K), v.shape == (B, H, N, K, D)
            v_xyz = self.linear2(xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)
            # v_xyz.shape == (B, H, N, K, D)
            # print('v_xyz.shape == (B, H, N, K, D)', v_xyz.shape)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
            # x.shape == (B, N, H, D)
        return x

    def local_attention_scalarSub(self, q, k, v, xyz):
        if self.pe_method == 'pe_ii':
            energy_xyz = self.linear1(xyz).permute(0, 3, 1, 2)  # energy_xyz.shape == (B, N, K, C=H)
            energy_xyz = energy_xyz.unsqueeze(3)  # energy_xyz.shape == (B, H, N, 1, K)
            # print('sub energy_xyz', energy_xyz.shape)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D) k.shape == (B, H, N, D, K)
            diff = q_repeated - k.permute(0, 1, 2, 4, 3)  # diff.shape == (B, H, N, K, D)
            energy = q @ diff.permute(0, 1, 2, 4, 3) + energy_xyz  # energy.shape == (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)
            # v_xyz.shape == (B, H, N, K, D)
            # print('sub v_xyz.shape == (B, H, N, K, D)', v_xyz.shape)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, 1, D)
            x = x[:, :, :, 0, :]  # x.shape == (B, N, H, D)
        return x

    def local_attention_scalarAdd(self, q, k, v, xyz):
        if self.pe_method == 'pe_ii':
            energy_xyz = self.linear1(xyz).permute(0, 3, 1, 2)  # energy_xyz.shape == (B, N, K, C=H)
            energy_xyz = energy_xyz.unsqueeze(3)  # energy_xyz.shape == (B, H, N, 1, K)
            # print('add energy_xyz', energy_xyz.shape)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
            energy = q_repeated + k.permute(0, 1, 2, 4, 3)  # energy.shape == (B, H, N, K, D)
            # scale_factor = math.sqrt(q.shape[-1])
            energy = torch.tanh(energy)  # attention.shape == (B, H, N, K, D)
            energy = energy @ self.p_add  # attention.shape == (B, H, N, K, 1)
            energy = energy.permute(0, 1, 2, 4, 3) + energy_xyz  # attention.shape == (B, H, N, 1, K)
            attention = self.softmax(energy)  # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)
            # v_xyz.shape == (B, H, N, K, D)
            # print('add v_xyz.shape == (B, H, N, K, D)', v_xyz.shape)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)
        return x

    def local_attention_scalarCat(self, q, k, v, xyz):
        if self.pe_method == 'pe_ii':
            energy_xyz = self.linear1(xyz).permute(0, 3, 1, 2)  # energy_xyz.shape == (B, N, K, C=H)
            energy_xyz = energy_xyz.unsqueeze(3)  # energy_xyz.shape == (B, H, N, 1, K)
            # print('cat energy_xyz', energy_xyz.shape)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)
            # q_repeated.shape == (B, H, N, K, D)
            energy = torch.cat((q_repeated, k.permute(0, 1, 2, 4, 3)), dim=-1)
            # energy.shape == (B, H, N, K, 2D)
            # scale_factor = math.sqrt(q.shape[-1])
            energy = torch.tanh(energy)
            # attention.shape == (B, H, N, K, 2D)
            energy = energy @ self.p_cat  # attention.shape == (B, H, N, K, 1)
            energy = energy.permute(0, 1, 2, 4, 3) + energy_xyz
            attention = self.softmax(energy)
            # attention.shape == (B, H, N, 1, K)
            v_xyz = self.linear2(xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)
            # v_xyz.shape == (B, H, N, K, D)
            # print('cat v_xyz.shape == (B, H, N, K, D)', v_xyz.shape)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
            # x.shape == (B, N, H, D)
        return x

    def local_attention_vectorSub(self, q, k, v, xyz):
        if self.pe_method == 'pe_ii':
            energy_xyz = self.linear1(xyz)  # v_xyz.shape == (B, N, K, C)
            energy_xyz = energy_xyz.view(energy_xyz.shape[0], self.num_heads, energy_xyz.shape[1], energy_xyz.shape[2], self.k_depth)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)
            # q_repeated.shape == (B, H, N, K, D) to match with k
            energy = q_repeated - k.permute(0, 1, 2, 4, 3) + energy_xyz
            # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)
            # attention.shape == (B, H, N, K, D)
            v_xyz = self.linear2(xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)
            # v_xyz.shape == (B, H, N, K, D)
            # print('vectorsub v_xyz.shape == (B, H, N, K, D)', v_xyz.shape)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)
            # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)
            # x.shape == (B, N, H, D)
        return x

    def local_attention_vectorAdd(self, q, k, v, xyz):
        if self.pe_method == 'pe_ii':
            energy_xyz = self.linear1(xyz)  # v_xyz.shape == (B, N, K, C)
            energy_xyz = energy_xyz.view(energy_xyz.shape[0], self.num_heads, energy_xyz.shape[1], energy_xyz.shape[2],
                                         self.k_depth)
            q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, 1, D)
            energy = q_repeated + k.permute(0, 1, 2, 4, 3) + energy_xyz # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, K, D)
            v_xyz = self.linear2(xyz)  # v_xyz.shape == (B, N, K, C)
            v_xyz = v_xyz.view(v_xyz.shape[0], self.num_heads, v_xyz.shape[1], v_xyz.shape[2], self.k_depth)
            # v_xyz.shape == (B, H, N, K, D)
            # print('vectoradd v_xyz.shape == (B, H, N, K, D)', v_xyz.shape)
            v = v + v_xyz  # v.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)
            # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)  # x.shape == (B, N, H, D)
        return x


"""
Global based attention variation input shape：
q.shape == (B, H, N, D)
k.shape == (B, H, N, D)
v.shape == (B, H, N, D)
"""


class Pe_global_based_attention_variation(nn.Module):
    def __init__(self, ):
        super(Pe_global_based_attention_variation, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return x

    def global_attention_Dot(self, q, k, v):
        energy = q @ k.permute(0, 1, 3, 2)  # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, N)
        x = (attention @ v).permute(0, 2, 1, 3)
        # x.shape == (B, N, H, D)
        return x

    def global_attention_Sub(self, q, k, v):
        inner = -2 * torch.matmul(q, k.transpose(3, 2))  # inner.shape == (B, H, N, N)
        qq = torch.sum(q ** 2, dim=-1, keepdim=True)  # qq.shape == (B, H, N, 1)
        kk = torch.sum(k ** 2, dim=-1, keepdim=True)  # kk.shape == (B, H, N, 1)
        diff_squared = qq + inner + kk.transpose(3, 2)  # pairwise_distance.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(-diff_squared / scale_factor)
        # attention.shape == (B, H, N, N)
        x = (attention @ v).permute(0, 2, 1, 3)
        # x.shape == (B, N, H, D)
        return x
