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


class Local_based_attention_variation(nn.Module):
    def __init__(self, k_out=64, num_heads=8, att_score_method='local_scalar_dot'):
        super(Local_based_attention_variation, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

        if att_score_method == 'local_scalar_add':
            self.k_depth = int(k_out / num_heads)
            self.p_add = nn.Parameter(torch.ones(self.k_depth, 1))  # p_add.shape == (D, 1)
        elif att_score_method == 'local_scalar_cat':
            self.k_depth = int(k_out / num_heads)
            self.p_cat = nn.Parameter(torch.ones(2 * self.k_depth, 1))  # p_cat.shape == (1, 2D)

    def local_attention_scalarDot(self, q, k, v):
        energy = q @ k
        # energy.shape == (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, 1, K)
        x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # x.shape == (B, N, H, D)
        return x

    def local_attention_scalarSub(self, q, k, v):
        q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D) k.shape == (B, H, N, D, K)
        diff = q_repeated - k.permute(0, 1, 2, 4, 3)  # diff.shape == (B, H, N, K, D)
        energy = q @ diff.permute(0, 1, 2, 4, 3)  # energy.shape == (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K)
        x = (attention @ v).permute(0, 2, 1, 3, 4)  # x.shape == (B, N, H, 1, D)
        x = x[:, :, :, 0, :]  # x.shape == (B, N, H, D)
        return x

    def local_attention_scalarAdd(self, q, k, v):
        q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)  # q_repeated.shape == (B, H, N, K, D)
        energy = q_repeated + k.permute(0, 1, 2, 4, 3)  # energy.shape == (B, H, N, K, D)
        # scale_factor = math.sqrt(q.shape[-1])
        attention = torch.tanh(energy)  # attention.shape == (B, H, N, K, D)
        attention = attention @ self.p_add  # attention.shape == (B, H, N, K, 1)
        attention = self.softmax(attention.permute(0, 1, 2, 4, 3))  # attention.shape == (B, H, N, 1, K)
        x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)  # x.shape == (B, N, H, D)

        return x

    def local_attention_scalarCat(self, q, k, v):
        q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)
        # q_repeated.shape == (B, H, N, K, D)
        energy = torch.cat((q_repeated, k.permute(0, 1, 2, 4, 3)), dim=-1)
        # energy.shape == (B, H, N, K, 2D)
        # scale_factor = math.sqrt(q.shape[-1])
        attention = torch.tanh(energy)
        # attention.shape == (B, H, N, K, 2D)
        attention = attention @ self.p_cat  # attention.shape == (B, H, N, K, 1)
        attention = self.softmax(attention.permute(0, 1, 2, 4, 3))
        # attention.shape == (B, H, N, 1, K)
        x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # x.shape == (B, N, H, D)
        return x

    def local_attention_vectorSub(self, q, k, v):
        q_repeated = q.repeat(1, 1, 1, k.shape[-1], 1)
        # q_repeated.shape == (B, H, N, K, D) to match with k
        energy = q_repeated - k.permute(0, 1, 2, 4, 3)
        # energy.shape == (B, H, N, K, D)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, K, D)
        x = (attention * v).permute(0, 2, 1, 3, 4)
        # x.shape == (B, N, H, K, D) element-wise multiplication
        x = x.sum(dim=-2)
        # x.shape == (B, N, H, D)
        return x

    def local_attention_vectorAdd(self, q, k, v):
        q_repeated = q.repeat(1, 1, 1,  k.shape[-1], 1)  # q_repeated.shape == (B, H, N, 1, D)
        energy = q_repeated + k.permute(0, 1, 2, 4, 3)  # energy.shape == (B, H, N, K, D)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, K, D)
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

class Global_based_attention_variation(nn.Module):
    def __init__(self, ):
        super(Global_based_attention_variation, self).__init__()
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