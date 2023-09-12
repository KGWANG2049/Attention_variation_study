import torch
from torch import nn
from utils import ops
import math
from models.attention_variants import Local_based_attention_variation, Global_based_attention_variation


class SelfAttention(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8,
                 att_score_method='global_dot'):
        super(SelfAttention, self).__init__()
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
        self.Global_based_attention_variation = Global_based_attention_variation()
        self.att_score_method = att_score_method
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.global_split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, D)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        k = self.global_split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, D)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        v = self.global_split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, D)

        if self.att_score_method == 'global_dot':
            x = self.Global_based_attention_variation.global_attention_Dot(q, k, v)

        elif self.att_score_method == 'global_sub':
            x = self.Global_based_attention_variation.global_attention_Sub(q, k, v)
            print("succeed using glosub", x.shape)
        else:
            raise ValueError(f"att_score_method must be 'global_dot', 'global_sub'. Got: {self.att_score_method}")

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


class Global_SelfAttention_Layer(nn.Module):
    def __init__(self, q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method,
                 ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out):
        super(Global_SelfAttention_Layer, self).__init__()

        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')
        self.att_score_method = att_score_method
        self.sa = SelfAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method)
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, pcd):
        x_out = self.sa(pcd)  # x_out.shape == (B, C, N)
        x = self.bn1(pcd + x_out)  # x.shape == (B, C, N)
        x_out = self.ff(x)  # x_out.shape == (B, C, N)
        x = self.bn2(x + x_out)  # x.shape == (B, C, N)
        return x


class CrossAttention(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8,
                 att_score_method='local_scalar_dot', neighbor_type='diff'):
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
        self.att_score_method = att_score_method
        self.neighbor_type = neighbor_type
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.Local_based_attention_variation = Local_based_attention_variation(k_out, num_heads, att_score_method)
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

        if self.att_score_method == 'local_scalar_dot':
            x = self.Local_based_attention_variation.local_attention_scalarDot(q, k, v)
            # print("8 2048 4 32", x.shape)
        elif self.att_score_method == 'local_scalar_sub':
            x = self.Local_based_attention_variation.local_attention_scalarSub(q, k, v)

        elif self.att_score_method == 'local_scalar_add':
            x = self.Local_based_attention_variation.local_attention_scalarAdd(q, k, v)

        elif self.att_score_method == 'local_scalar_cat':
            x = self.Local_based_attention_variation.local_attention_scalarCat(q, k, v)

        elif self.att_score_method == 'local_vector_sub':
            x = self.Local_based_attention_variation.local_attention_vectorSub(q, k, v)

        elif self.att_score_method == 'local_vector_add':
            x = self.Local_based_attention_variation.local_attention_vectorAdd(q, k, v)

        else:
            raise ValueError(f'Invalid value for att_score_method: {self.att_score_method}')

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


class Local_CrossAttention_Layer(nn.Module):
    def __init__(self, single_scale_or_multi_scale, key_one_or_sep, shared_ca, K, scale, neighbor_selection_method,
                 neighbor_type, mlp_or_sum,
                 q_in, q_out, k_in, k_out, v_in,
                 v_out, num_heads, att_score_method, ff_conv1_channels_in,
                 ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out):
        super(Local_CrossAttention_Layer, self).__init__()

        self.single_scale_or_multi_scale = single_scale_or_multi_scale
        self.key_one_or_sep = key_one_or_sep
        self.shared_ca = shared_ca
        self.K = K
        self.scale = scale
        self.neighbor_selection_method = neighbor_selection_method
        self.neighbor_type = neighbor_type
        self.mlp_or_sum = mlp_or_sum

        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')

        if self.single_scale_or_multi_scale == 'ss':
            self.ca = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method, neighbor_type)

        elif self.single_scale_or_multi_scale == 'ms':

            if self.key_one_or_sep == 'one':
                self.ca = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method,
                                         neighbor_type)

            elif self.key_one_or_sep == 'sep':

                if self.shared_ca:
                    self.ca = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method,
                                             neighbor_type)
                else:
                    self.ca_list = nn.ModuleList(
                        [CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method,
                                        neighbor_type) for _ in range(scale + 1)])

                if self.mlp_or_sum == 'mlp':
                    self.linear = nn.Conv1d(v_out * (scale + 1), q_in, 1, bias=False)

            else:
                raise ValueError(f'key_one_or_sep should be one or sep! but got {self.key_one_or_sep}')

        else:
            raise ValueError(
                f'single_scale_or_multi_scale should be ss or ms, but got {self.single_scale_or_multi_scale}')

        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, pcd, coordinate):
        if self.single_scale_or_multi_scale == 'ss':
            neighbors, idx_all = ops.select_neighbors_single_scale(pcd, coordinate, self.K, self.scale,
                                                                   self.neighbor_selection_method,
                                                                   self.neighbor_type)
            neighbors = neighbors.permute(0, 3, 1, 2)  # neighbor.shape == (B, num x C, N, K)
            x_out = self.ca(pcd, neighbors)  # x_out.shape == (B, C, N)
        elif self.single_scale_or_multi_scale == 'ms':

            if self.key_one_or_sep == 'one':
                neighbors, idx_all = ops.select_neighbors_in_one_key(pcd, coordinate, self.K, self.scale,
                                                                     self.neighbor_selection_method, self.neighbor_type)
                neighbors = neighbors.permute(0, 3, 1, 2)  # neighbor.shape == (B, num x C, N, (scale+1) x K)
                x_out = self.ca(pcd, neighbors)  # x_out.shape == (B, C, N)
            elif self.key_one_or_sep == 'sep':
                neighbors, idx_all = ops.select_neighbors_in_separate_key(pcd, coordinate, self.K, self.scale,
                                                                          self.neighbor_selection_method,
                                                                          self.neighbor_type)
                neighbors = neighbors.permute(0, 3, 1, 2)  # neighbor.shape == (B, num x C, N, (scale+1) x K)
                neighbor_list = ops.list_generator(neighbors, self.K, self.scale)  # element shape in list == (B,
                # num x C, N, K)
                # print("neighbor_list", neighbor_list.shape)
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
                        # print("neighbors", neighbors.shape)
                        x_out = ca(pcd, neighbors)
                        # x_out.shape == (B, C, N)
                        x_output_list.append(x_out)

                if self.mlp_or_sum == 'mlp':
                    x_out = torch.cat(x_output_list, dim=1)  # x_out.shape == (B, (scale + 1) x C, N)
                    x_out = self.linear(x_out)  # x_out.shape == (B, C, N)
                elif self.mlp_or_sum == 'sum':
                    x_out = torch.stack(x_output_list)  # x_out.shape == (scale+1, B, C, N)
                    x_out = torch.sum(x_out, dim=0)  # x_out.shape == (B, C, N)

        # x_out.shape == (B, C, N)
        # print("pcd", pcd.shape, "x_out", x_out.shape)
        x = self.bn1(pcd + x_out)
        # x.shape == (B, C, N)
        x_out = self.ff(x)
        # x_out.shape == (B, C, N)
        x = self.bn2(x + x_out)
        # x.shape == (B, C, N)
        return x


class Point_Embedding(nn.Module):
    def __init__(self, embedding_k, conv1_channel_in, conv1_channel_out, point_emb2_in, point_emb2_out):
        super(Point_Embedding, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(conv1_channel_in, conv1_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv1_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(point_emb2_in, point_emb2_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(point_emb2_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.embedding_k = embedding_k

    # init_a.shape == (B, 3, N)
    # init_b.shape == (B, 3, N)
    def forward(self, a):
        # print("a initial shape:", a.shape)
        # print("b initial shape:", b.shape)
        a = a.permute(0, 2, 1)  # a.shape == (B, N, 3)
        # print("a after permute shape:", a.shape)
        # print("b after permute shape:", b.shape)
        idx = ops.knn(a, a, self.embedding_k)  # idx.shape == (B, N, K)
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


class ModelNetModelCls(nn.Module):
    def __init__(self, embedding_k, point_emb1_in, point_emb1_out, point_emb2_in, point_emb2_out,
                 global_or_local, single_scale_or_multi_scale, key_one_or_sep, shared_ca, K, scale,
                 neighbor_selection_method,
                 neighbor_type, mlp_or_sum,
                 q_in, q_out, k_in, k_out, v_in,
                 v_out, num_heads, att_score_method, ff_conv1_channels_in,
                 ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out):
        super(ModelNetModelCls, self).__init__()
        self.k_out = k_out[0]
        self.num_att_layer = len(k_out)
        self.Point_Embedding_list = nn.ModuleList(
            [Point_Embedding(embedding_k, point_emb1_in, point_emb1_out, point_emb2_in, point_emb2_out) for
             embedding_k, point_emb1_in, point_emb1_out, point_emb2_in, point_emb2_out in
             zip(embedding_k, point_emb1_in, point_emb1_out, point_emb2_in, point_emb2_out)])

        self.global_or_local = global_or_local
        if self.global_or_local == 'global':
            self.Global_SelfAttention_Layer_list = nn.ModuleList(
                [Global_SelfAttention_Layer(q_in, q_out, k_in, k_out, v_in,
                                            v_out, num_heads, att_score_method,
                                            ff_conv1_channels_in,
                                            ff_conv1_channels_out,
                                            ff_conv2_channels_in,
                                            ff_conv2_channels_out) for
                 q_in, q_out, k_in, k_out, v_in, v_out, num_heads,
                 att_score_method, ff_conv1_channels_in,
                 ff_conv1_channels_out, ff_conv2_channels_in,
                 ff_conv2_channels_out
                 in
                 zip(q_in, q_out, k_in, k_out, v_in, v_out, num_heads,
                     att_score_method, ff_conv1_channels_in,
                     ff_conv1_channels_out, ff_conv2_channels_in,
                     ff_conv2_channels_out)])
        elif self.global_or_local == 'local':
            self.Local_CrossAttention_Layer_list = nn.ModuleList(
                [Local_CrossAttention_Layer(single_scale_or_multi_scale, key_one_or_sep, shared_ca, K, scale,
                                            neighbor_selection_method,
                                            neighbor_type, mlp_or_sum,
                                            q_in, q_out, k_in, k_out, v_in,
                                            v_out, num_heads, att_score_method, ff_conv1_channels_in,
                                            ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out) for
                 single_scale_or_multi_scale, key_one_or_sep, shared_ca, K, scale, neighbor_selection_method,
                 neighbor_type, mlp_or_sum,
                 q_in, q_out, k_in, k_out, v_in,
                 v_out, num_heads, att_score_method, ff_conv1_channels_in,
                 ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out in
                 zip(single_scale_or_multi_scale, key_one_or_sep, shared_ca, K, scale, neighbor_selection_method,
                     neighbor_type, mlp_or_sum,
                     q_in, q_out, k_in, k_out, v_in,
                     v_out, num_heads, att_score_method, ff_conv1_channels_in,
                     ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)])

        self.linear0 = nn.Sequential(nn.Conv1d(self.k_out * self.num_att_layer, 1024, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(1024),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))
        self.linear3 = nn.Linear(256, 40)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

        # self.conv_list = nn.ModuleList(
        # [nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False) for channel_in in ff_conv2_channels_out])

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
            x = point_embedding(x)  # x.shape == (B, C, N)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)  # x.shape == (B, num_layer x C, N)

        if self.global_or_local == 'global':
            for Global_SelfAttention_Layer in self.Global_SelfAttention_Layer_list:
                x = Global_SelfAttention_Layer(x)
                res_link_list.append(x)
        elif self.global_or_local == 'local':
            # i = 0
            for Local_CrossAttention_Layer in self.Local_CrossAttention_Layer_list:
                x = Local_CrossAttention_Layer(x, xyz)
                # x.shape == (B, C, N)
                # x_extract = x.max(dim=-1)[0]
                # x_expand = self.conv_list[i](x).max(dim=-1)[0]  # x_expand.shape == (B, 1024)
                res_link_list.append(x)
                # print("x.shape", x.shape)
                # i += 1
        x = torch.cat(res_link_list, dim=1)  # x.shape == (B, 4096)  or  (B, 512, N)
        x = self.linear0(x)  # x.shape == (B, 1024, N)
        x = x.max(dim=-1)[0]  # x.shape == (B, 1024)
        x = self.linear1(x)
        # x.shape == (B, 1024)
        x = self.dp1(x)
        # x.shape == (B, 1024)
        x = self.linear2(x)
        # x.shape == (B, 512)
        x = self.dp2(x)
        # x.shape == (B, 256)
        x = self.linear3(x)
        # x.shape == (B, 40)
        return x


# about local attention with position encoding
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

# about global attention with positon encoding
"""
    if att_score_method == 'dot_product':
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

    elif att_score_method == 'subtraction':
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

    elif att_score_method == 'addition':
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
"""
