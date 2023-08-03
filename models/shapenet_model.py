import torch
from torch import nn
from utils import ops
import math
import torch.nn.functional as F


class EdgeConvBlock(nn.Module):
    def __init__(self, egdeconv_emb_K=40, egdeconv_emb_group_type='center_diff',
                 egdeconv_emb_conv1_in=6, egdeconv_emb_conv1_out=64, egdeconv_emb_conv2_in=64,
                 egdeconv_emb_conv2_out=64,
                 downsample_which='p2p', downsample_k=(1024, 512), downsample_q_in=(64, 64), downsample_q_out=(64, 64),
                 downsample_k_in=(64, 64), downsample_k_out=(64, 64), downsample_v_in=(64, 64),
                 downsample_v_out=(64, 64), downsample_num_heads=(1, 1),
                 upsample_which='crossA', upsample_q_in=(64, 64), upsample_q_out=(64, 64),
                 upsample_k_in=(64, 64), upsample_k_out=(64, 64), upsample_v_in=(64, 64),
                 upsample_v_out=(64, 64), upsample_num_heads=(1, 1),
                 K=(32, 32, 32), group_type=('center_diff', 'center_diff', 'center_diff'),
                 conv1_channel_in=(3 * 2, 64 * 2, 64 * 2), conv1_channel_out=(64, 64, 64),
                 conv2_channel_in=(64, 64, 64), conv2_channel_out=(64, 64, 64)):
        super(EdgeConvBlock, self).__init__()
        self.embedding_list = nn.ModuleList(
            [EdgeConv(emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out) for
             emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in
             zip(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out,
                 egdeconv_emb_conv2_in, egdeconv_emb_conv2_out)])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList(
                [DownSample(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for
                 ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in
                 zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out,
                     downsample_v_in, downsample_v_out, downsample_num_heads)])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList(
                [DownSampleWithSigma(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for
                 ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in
                 zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out,
                     downsample_v_in, downsample_v_out, downsample_num_heads)])
        else:
            raise ValueError('Only global and local are valid for which_ds!')
        if upsample_which == 'crossA':
            self.upsample_list = nn.ModuleList(
                [UpSample(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for
                 us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in
                 zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out,
                     upsample_num_heads)])
        elif upsample_which == 'selfA':
            self.upsample_list = nn.ModuleList(
                [UpSampleSelfAttention(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for
                 us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in
                 zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out,
                     upsample_num_heads)])
        else:
            raise ValueError('Only crossA and selfA are valid for which_ups!')
        self.edgeconv_list = nn.ModuleList([EdgeConv(k, g_type, conv1_in, conv1_out, conv2_in, conv2_out) for
                                            k, g_type, conv1_in, conv1_out, conv2_in, conv2_out in
                                            zip(K, group_type, conv1_channel_in, conv1_channel_out, conv2_channel_in,
                                                conv2_channel_out)])

    def forward(self, x):
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.edgeconv_list[0](x)
        x_list = [x]
        points_drop_list = []
        idx_select_list = []
        idx_drop_list = []
        for i in range(len(self.downsample_list)):
            (x, idx_select), (points_drop, idx_drop) = self.downsample_list[i](x)
            x = self.edgeconv_list[i + 1](x)
            x_list.append(x)
            points_drop_list.append(points_drop)
            idx_select_list.append(idx_select)
            idx_drop_list.append(idx_drop)
        split = int((len(self.edgeconv_list) - 1) / 2)
        x = ((x_list.pop(), idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        for j in range(len(self.upsample_list)):
            x_tmp = x_list.pop()
            x = self.upsample_list[j](x_tmp, x)
            x = self.edgeconv_list[j + 1 + split](x)
            if j < len(self.upsample_list) - 1:
                x = ((x, idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        return x


class EdgeConv(nn.Module):
    def __init__(self, K=32, group_type='center_diff', conv1_channel_in=6, conv1_channel_out=64, conv2_channel_in=64,
                 conv2_channel_out=64):
        super(EdgeConv, self).__init__()
        self.K = K
        self.group_type = group_type

        self.conv1 = nn.Sequential(nn.Conv2d(conv1_channel_in, conv1_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv1_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv2_channel_in, conv2_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv2_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # x.shape == (B, C, N)
        x = ops.group(x, self.K, self.group_type)
        # x.shape == (B, 2C, N, K) or (B, C, N, K)
        x = self.conv1(x)
        # x.shape == (B, C, N, K)
        x = self.conv2(x)
        # x.shape == (B, C, N, K)
        x = x.max(dim=-1, keepdim=False)[0]
        # x.shape == (B, C, N)
        return x


class Neighbor2PointAttentionBlock(nn.Module):
    def __init__(self, egdeconv_emb_K=40, egdeconv_emb_group_type='center_diff',
                 egdeconv_emb_conv1_in=6, egdeconv_emb_conv1_out=64, egdeconv_emb_conv2_in=64,
                 egdeconv_emb_conv2_out=64,
                 downsample_which='p2p', downsample_k=(1024, 512), downsample_q_in=(64, 64), downsample_q_out=(64, 64),
                 downsample_k_in=(64, 64), downsample_k_out=(64, 64), downsample_v_in=(64, 64),
                 downsample_v_out=(64, 64), downsample_num_heads=(1, 1),
                 upsample_which='crossA', upsample_q_in=(64, 64), upsample_q_out=(64, 64),
                 upsample_k_in=(64, 64), upsample_k_out=(64, 64), upsample_v_in=(64, 64),
                 upsample_v_out=(64, 64), upsample_num_heads=(1, 1),
                 K=(32, 32, 32), group_type=('diff', 'diff', 'diff'), q_in=(64, 64, 64), q_out=(64, 64, 64),
                 k_in=(64, 64, 64), k_out=(64, 64, 64), v_in=(64, 64, 64), v_out=(64, 64, 64), num_heads=(8, 8, 8),
                 ff_conv1_channels_in=(64, 64, 64), ff_conv1_channels_out=(128, 128, 128),
                 ff_conv2_channels_in=(128, 128, 128), ff_conv2_channels_out=(64, 64, 64)):
        super(Neighbor2PointAttentionBlock, self).__init__()
        self.embedding_list = nn.ModuleList(
            [EdgeConv(emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out) for
             emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in
             zip(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out,
                 egdeconv_emb_conv2_in, egdeconv_emb_conv2_out)])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList(
                [DownSample(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for
                 ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in
                 zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out,
                     downsample_v_in, downsample_v_out, downsample_num_heads)])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList(
                [DownSampleWithSigma(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for
                 ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in
                 zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out,
                     downsample_v_in, downsample_v_out, downsample_num_heads)])
        else:
            raise ValueError('Only global and local are valid for which_ds!')
        if upsample_which == 'crossA':
            self.upsample_list = nn.ModuleList(
                [UpSample(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for
                 us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in
                 zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out,
                     upsample_num_heads)])
        elif upsample_which == 'selfA':
            self.upsample_list = nn.ModuleList(
                [UpSampleSelfAttention(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for
                 us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in
                 zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out,
                     upsample_num_heads)])
        else:
            raise ValueError('Only crossA and selfA are valid for which_ups!')
        self.neighbor2point_list = nn.ModuleList([Neighbor2PointAttention(k, g_type, q_input, q_output, k_input,
                                                                          k_output, v_input, v_output, heads,
                                                                          ff_conv1_channel_in, ff_conv1_channel_out,
                                                                          ff_conv2_channel_in, ff_conv2_channel_out)
                                                  for
                                                  k, g_type, q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out
                                                  in
                                                  zip(K, group_type, q_in, q_out, k_in, k_out, v_in, v_out, num_heads,
                                                      ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in,
                                                      ff_conv2_channels_out)])

    def forward(self, x):
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.neighbor2point_list[0](x)
        x_list = [x]
        points_drop_list = []
        idx_select_list = []
        idx_drop_list = []
        for i in range(len(self.downsample_list)):
            (x, idx_select), (points_drop, idx_drop) = self.downsample_list[i](x)
            x = self.neighbor2point_list[i + 1](x)
            x_list.append(x)
            points_drop_list.append(points_drop)
            idx_select_list.append(idx_select)
            idx_drop_list.append(idx_drop)
        split = int((len(self.neighbor2point_list) - 1) / 2)
        x = ((x_list.pop(), idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        for j in range(len(self.upsample_list)):
            x_tmp = x_list.pop()
            x = self.upsample_list[j](x_tmp, x)
            x = self.neighbor2point_list[j + 1 + split](x)
            if j < len(self.upsample_list) - 1:
                x = ((x, idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        return x


class Neighbor2PointAttention(nn.Module):
    def __init__(self, K=32, group_type='diff', q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8,
                 ff_conv1_channels_in=64, ff_conv1_channels_out=128, ff_conv2_channels_in=128,
                 ff_conv2_channels_out=64):
        super(Neighbor2PointAttention, self).__init__()
        # check input values
        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.K = K
        self.group_type = group_type
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, x):
        # x.shape == (B, C, N)
        neighbors = ops.group(x, self.K, self.group_type)
        # neighbors.shape == (B, C, N, K)
        x_tmp = x[:, :, :, None]
        # x_tmp.shape == (B, C, N, 1)
        q = self.q_conv(x_tmp)
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
        energy = q @ k
        # energy.shape == (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, 1, K)
        x_tmp = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # x_tmp.shape == (B, N, H, D)
        x_tmp = x_tmp.reshape(x_tmp.shape[0], x_tmp.shape[1], -1).permute(0, 2, 1)
        # x_tmp.shape == (B, C, N)
        x = self.bn1(x + x_tmp)
        # x.shape == (B, C, N)
        x_tmp = self.ff(x)
        # x_tmp.shape == (B, C, N)
        x = self.bn2(x + x_tmp)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x


class Point2PointAttentionBlock(nn.Module):
    def __init__(self, egdeconv_emb_K=40, egdeconv_emb_group_type='center_diff',
                 egdeconv_emb_conv1_in=6, egdeconv_emb_conv1_out=64, egdeconv_emb_conv2_in=64,
                 egdeconv_emb_conv2_out=64,
                 downsample_which='p2p', downsample_k=(1024, 512), downsample_q_in=(64, 64), downsample_q_out=(64, 64),
                 downsample_k_in=(64, 64), downsample_k_out=(64, 64), downsample_v_in=(64, 64),
                 downsample_v_out=(64, 64), downsample_num_heads=(1, 1),
                 upsample_which='crossA', upsample_q_in=(64, 64), upsample_q_out=(64, 64),
                 upsample_k_in=(64, 64), upsample_k_out=(64, 64), upsample_v_in=(64, 64),
                 upsample_v_out=(64, 64), upsample_num_heads=(1, 1),
                 q_in=(64, 64, 64), q_out=(64, 64, 64), k_in=(64, 64, 64), k_out=(64, 64, 64), v_in=(64, 64, 64),
                 v_out=(64, 64, 64), num_heads=(8, 8, 8),
                 ff_conv1_channels_in=(64, 64, 64), ff_conv1_channels_out=(128, 128, 128),
                 ff_conv2_channels_in=(128, 128, 128), ff_conv2_channels_out=(64, 64, 64)):
        super(Point2PointAttentionBlock, self).__init__()
        self.embedding_list = nn.ModuleList(
            [EdgeConv(emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out) for
             emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in
             zip(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out,
                 egdeconv_emb_conv2_in, egdeconv_emb_conv2_out)])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList(
                [DownSample(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for
                 ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in
                 zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out,
                     downsample_v_in, downsample_v_out, downsample_num_heads)])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList(
                [DownSampleWithSigma(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for
                 ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in
                 zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out,
                     downsample_v_in, downsample_v_out, downsample_num_heads)])
        else:
            raise ValueError('Only global and local are valid for which_ds!')
        if upsample_which == 'crossA':
            self.upsample_list = nn.ModuleList(
                [UpSample(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for
                 us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in
                 zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out,
                     upsample_num_heads)])
        elif upsample_which == 'selfA':
            self.upsample_list = nn.ModuleList(
                [UpSampleSelfAttention(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for
                 us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in
                 zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out,
                     upsample_num_heads)])
        else:
            raise ValueError('Only crossA and selfA are valid for which_ups!')
        self.point2point_list = nn.ModuleList([Point2PointAttention(q_input, q_output, k_input, k_output, v_input,
                                                                    v_output, heads, ff_conv1_channel_in,
                                                                    ff_conv1_channel_out, ff_conv2_channel_in,
                                                                    ff_conv2_channel_out)
                                               for
                                               q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out
                                               in zip(q_in, q_out, k_in, k_out, v_in, v_out, num_heads,
                                                      ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in,
                                                      ff_conv2_channels_out)])

    def forward(self, x):
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.point2point_list[0](x)
        x_list = [x]
        points_drop_list = []
        idx_select_list = []
        idx_drop_list = []
        for i in range(len(self.downsample_list)):
            (x, idx_select), (points_drop, idx_drop) = self.downsample_list[i](x)
            x = self.point2point_list[i + 1](x)
            x_list.append(x)
            points_drop_list.append(points_drop)
            idx_select_list.append(idx_select)
            idx_drop_list.append(idx_drop)
        split = int((len(self.point2point_list) - 1) / 2)
        x = ((x_list.pop(), idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        for j in range(len(self.upsample_list)):
            x_tmp = x_list.pop()
            x = self.upsample_list[j](x_tmp, x)
            x = self.point2point_list[j + 1 + split](x)
            if j < len(self.upsample_list) - 1:
                x = ((x, idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        return x


class Point2PointAttention(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8,
                 ff_conv1_channels_in=64, ff_conv1_channels_out=128,
                 ff_conv2_channels_in=128, ff_conv2_channels_out=64):
        super(Point2PointAttention, self).__init__()
        # check input values
        if q_in != k_in or q_in != v_in or k_in != v_in:
            raise ValueError(f'q_in, k_in and v_in should be the same! Got q_in:{q_in}, k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, x):
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, N)
        x_tmp = (attention @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # x_tmp.shape == (B, N, H, D)
        x_tmp = x_tmp.reshape(x_tmp.shape[0], x_tmp.shape[1], -1).permute(0, 2, 1)
        # x_tmp.shape == (B, C, N)
        x = self.bn1(x + x_tmp)
        # x.shape == (B, C, N)
        x_tmp = self.ff(x)
        # x_tmp.shape == (B, C, N)
        x = self.bn2(x + x_tmp)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x


class DownSample(nn.Module):
    def __init__(self, k, q_in, q_out, k_in, k_out, v_in, v_out, num_heads):
        super(DownSample, self).__init__()
        # check input values
        if q_in != k_in or q_in != v_in or k_in != v_in:
            raise ValueError(f'q_in, k_in and v_in should be the same! Got q_in:{q_in}, k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.k = k
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
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, N)
        self.idx = torch.sum(attention, dim=-2).topk(self.k, dim=-1)[1]
        # idx.shape == (B, H, K)
        idx_dropped = torch.sum(attention, dim=-2).topk(attention.shape[-1] - self.k, dim=-1, largest=False)[1]
        # idx_dropped.shape == (B, H, N-K)
        attention_down = torch.gather(attention, dim=2, index=self.idx[..., None].expand(-1, -1, -1, q.shape[-1]))
        # attention_down.shape == (B, H, K, N)
        attention_dropped = torch.gather(attention, dim=2, index=idx_dropped[..., None].expand(-1, -1, -1, q.shape[-1]))
        # attention_dropped.shape == (B, H, N-K, N)
        v_down = (attention_down @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # v_down.shape == (B, K, H, D)
        v_dropped = (attention_dropped @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # v_dropped.shape == (B, N-K, H, D)
        v_down = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)
        # v_down.shape == (B, C, K)
        v_dropped = v_dropped.reshape(v_dropped.shape[0], v_dropped.shape[1], -1).permute(0, 2, 1)
        # v_dropped.shape == (B, C, N-K)
        return (v_down, self.idx), (v_dropped, idx_dropped)

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x


class DownSampleWithSigma(nn.Module):
    def __init__(self, k, q_in, q_out, k_in, k_out, v_in, v_out, num_heads):
        super(DownSampleWithSigma, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.k = k  # number of downsampled points
        self.K = 32  # number of neighbors
        self.group_type = 'diff'
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape == (B, C, N)
        neighbors = ops.group(x, self.K, self.group_type)
        # neighbors.shape == (B, C, N, K)
        x = x[:, :, :, None]
        # x.shape == (B, C, N, 1)
        q = self.q_conv(x)
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
        energy = q @ k
        # energy.shape == (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, 1, K)
        self.idx = torch.std(attention, dim=-1, unbiased=False)[:, :, :, 0].topk(self.k, dim=-1)[1]
        # idx.shape == (B, H, M)
        idx_dropped = \
            torch.std(attention, dim=-1, unbiased=False)[:, :, :, 0].topk(attention.shape[-3] - self.k, dim=-1,
                                                                          largest=False)[1]
        # idx_dropped.shape == (B, H, N-M)
        attention_down = torch.gather(attention, dim=2,
                                      index=self.idx[..., None, None].expand(-1, -1, -1, -1, k.shape[-1]))
        # attention_down.shape == (B, H, M, 1, K)
        attention_dropped = torch.gather(attention, dim=2,
                                         index=idx_dropped[..., None, None].expand(-1, -1, -1, -1, k.shape[-1]))
        # attention_dropped.shape == (B, H, N-M, 1, K)
        v_down = torch.gather(v, dim=2, index=self.idx[..., None, None].expand(-1, -1, -1, k.shape[-1], k.shape[-2]))
        # v_down.shape == (B, H, M, K, D)
        v_dropped = torch.gather(v, dim=2,
                                 index=idx_dropped[..., None, None].expand(-1, -1, -1, k.shape[-1], k.shape[-2]))
        # v_dropped.shape == (B, H, N-M, K, D)
        v_down = (attention_down @ v_down)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # v_down.shape == (B, M, H, D)
        v_dropped = (attention_dropped @ v_dropped)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # v_dropped.shape == (B, N-M, H, D)
        v_down = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)
        # v_down.shape == (B, C, M)
        v_dropped = v_dropped.reshape(v_dropped.shape[0], v_dropped.shape[1], -1).permute(0, 2, 1)
        # v_dropped.shape == (B, C, N-M)
        return (v_down, self.idx), (v_dropped, idx_dropped)

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x


class UpSample(nn.Module):
    def __init__(self, q_in, q_out, k_in, k_out, v_in, v_out, num_heads):
        super(UpSample, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.skip_link = nn.Conv1d(q_in, v_out, 1, bias=False)

    def forward(self, pcd_up, pcd_down):
        (points_select, idx_select), (points_drop, idx_drop) = pcd_down
        # pcd_up.shape == (B, C, K1)  points_select.shape == (B, C, K2)
        q = self.q_conv(pcd_up)
        # q.shape == (B, C, K1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, K1)
        k = self.k_conv(points_select)
        # k.shape == (B, C, K2)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, K2)
        v = self.v_conv(points_select)
        # v.shape == (B, C, K2)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, K2)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, K1, K2)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, K1, K2)
        x = (attention @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # x.shape == (B, K1, H, D)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, K1)
        x = self.skip_link(pcd_up) + x
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x


class UpSampleSelfAttention(nn.Module):
    def __init__(self, q_in, q_out, k_in, k_out, v_in, v_out, num_heads):
        super(UpSampleSelfAttention, self).__init__()
        # check input values
        if q_in != k_in or q_in != v_in or k_in != v_in:
            raise ValueError(f'q_in, k_in and v_in should be the same! Got q_in:{q_in}, k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.skip_link = nn.Conv1d(q_in, v_out, 1, bias=False)

    def forward(self, pcd_up, pcd_down):
        (points_select, idx_select), (points_drop, idx_drop) = pcd_down
        # points_select.shape == (B, C, K1)  points_drop.shape == (B, C, K2)
        x = self.concat_by_idx(points_select, points_drop, idx_select, idx_drop, dim=-1)
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape == (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape == (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, N)
        x = (attention @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # x.shape == (B, N, H, D)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, N)
        x = self.skip_link(pcd_up) + x
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x

    def concat_by_idx(self, a, b, idx_a, idx_b, dim):
        src = torch.cat([a, b], dim=dim)
        target = torch.zeros_like(src)
        idx_tmp = torch.cat([idx_a, idx_b], dim=dim).expand(-1, a.shape[1], -1)
        return target.scatter(dim=-1, index=idx_tmp, src=src)


class ShapeNetModel(nn.Module):
    def __init__(self, neighbor2point_enable, neighbor2point_egdeconv_emb_K, neighbor2point_egdeconv_emb_group_type,
                 neighbor2point_egdeconv_emb_conv1_in,
                 neighbor2point_egdeconv_emb_conv1_out, neighbor2point_egdeconv_emb_conv2_in,
                 neighbor2point_egdeconv_emb_conv2_out,
                 neighbor2point_down_which, neighbor2point_down_k, neighbor2point_down_q_in, neighbor2point_down_q_out,
                 neighbor2point_down_k_in, neighbor2point_down_k_out, neighbor2point_down_v_in,
                 neighbor2point_down_v_out,
                 neighbor2point_down_num_heads, neighbor2point_up_which, neighbor2point_up_q_in,
                 neighbor2point_up_q_out, neighbor2point_up_k_in, neighbor2point_up_k_out, neighbor2point_up_v_in,
                 neighbor2point_up_v_out, neighbor2point_up_num_heads,
                 neighbor2point_K, neighbor2point_group_type, neighbor2point_q_in,
                 neighbor2point_q_out, neighbor2point_k_in, neighbor2point_k_out, neighbor2point_v_in,
                 neighbor2point_v_out, neighbor2point_num_heads,
                 neighbor2point_ff_conv1_in, neighbor2point_ff_conv1_out, neighbor2point_ff_conv2_in,
                 neighbor2point_ff_conv2_out,
                 point2point_enable, point2point_egdeconv_emb_K, point2point_egdeconv_emb_group_type,
                 point2point_egdeconv_emb_conv1_in, point2point_egdeconv_emb_conv1_out,
                 point2point_egdeconv_emb_conv2_in, point2point_egdeconv_emb_conv2_out,
                 point2point_down_which, point2point_down_k, point2point_down_q_in, point2point_down_q_out,
                 point2point_down_k_in,
                 point2point_down_k_out, point2point_down_v_in, point2point_down_v_out, point2point_down_num_heads,
                 point2point_up_which, point2point_up_q_in, point2point_up_q_out, point2point_up_k_in,
                 point2point_up_k_out, point2point_up_v_in, point2point_up_v_out, point2point_up_num_heads,
                 point2point_q_in, point2point_q_out, point2point_k_in, point2point_k_out, point2point_v_in,
                 point2point_v_out, point2point_num_heads, point2point_ff_conv1_in, point2point_ff_conv1_out,
                 point2point_ff_conv2_in, point2point_ff_conv2_out,
                 edgeconv_enable, egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in,
                 egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out, edgeconv_downsample_which,
                 edgeconv_downsample_k, edgeconv_downsample_q_in, edgeconv_downsample_q_out,
                 edgeconv_downsample_k_in, edgeconv_downsample_k_out, edgeconv_downsample_v_in,
                 edgeconv_downsample_v_out, edgeconv_downsample_num_heads,
                 edgeconv_upsample_which, edgeconv_upsample_q_in, edgeconv_upsample_q_out,
                 edgeconv_upsample_k_in, edgeconv_upsample_k_out, edgeconv_upsample_v_in,
                 edgeconv_upsample_v_out, edgeconv_upsample_num_heads,
                 edgeconv_K, edgeconv_group_type, edgeconv_conv1_channel_in, edgeconv_conv1_channel_out,
                 edgeconv_conv2_channel_in, edgeconv_conv2_channel_out):

        super(ShapeNetModel, self).__init__()

        num_enabled_blocks = neighbor2point_enable + point2point_enable + edgeconv_enable
        if num_enabled_blocks != 1:
            raise ValueError(
                f'Only one of neighbor2point_block, point2point_block and edgecov_block should be enabled, but got {num_enabled_blocks} block(s) enabled!')
        if neighbor2point_enable:
            self.block = Neighbor2PointAttentionBlock(neighbor2point_egdeconv_emb_K,
                                                      neighbor2point_egdeconv_emb_group_type,
                                                      neighbor2point_egdeconv_emb_conv1_in,
                                                      neighbor2point_egdeconv_emb_conv1_out,
                                                      neighbor2point_egdeconv_emb_conv2_in,
                                                      neighbor2point_egdeconv_emb_conv2_out,
                                                      neighbor2point_down_which, neighbor2point_down_k,
                                                      neighbor2point_down_q_in, neighbor2point_down_q_out,
                                                      neighbor2point_down_k_in, neighbor2point_down_k_out,
                                                      neighbor2point_down_v_in, neighbor2point_down_v_out,
                                                      neighbor2point_down_num_heads, neighbor2point_up_which,
                                                      neighbor2point_up_q_in, neighbor2point_up_q_out,
                                                      neighbor2point_up_k_in, neighbor2point_up_k_out,
                                                      neighbor2point_up_v_in, neighbor2point_up_v_out,
                                                      neighbor2point_up_num_heads,
                                                      neighbor2point_K, neighbor2point_group_type,
                                                      neighbor2point_q_in, neighbor2point_q_out, neighbor2point_k_in,
                                                      neighbor2point_k_out,
                                                      neighbor2point_v_in, neighbor2point_v_out,
                                                      neighbor2point_num_heads, neighbor2point_ff_conv1_in,
                                                      neighbor2point_ff_conv1_out, neighbor2point_ff_conv2_in,
                                                      neighbor2point_ff_conv2_out)
            output_channels = neighbor2point_ff_conv2_out[-1]
        if point2point_enable:
            self.block = Point2PointAttentionBlock(point2point_egdeconv_emb_K, point2point_egdeconv_emb_group_type,
                                                   point2point_egdeconv_emb_conv1_in,
                                                   point2point_egdeconv_emb_conv1_out,
                                                   point2point_egdeconv_emb_conv2_in,
                                                   point2point_egdeconv_emb_conv2_out,
                                                   point2point_down_which, point2point_down_k, point2point_down_q_in,
                                                   point2point_down_q_out, point2point_down_k_in,
                                                   point2point_down_k_out, point2point_down_v_in,
                                                   point2point_down_v_out, point2point_down_num_heads,
                                                   point2point_up_which, point2point_up_q_in, point2point_up_q_out,
                                                   point2point_up_k_in,
                                                   point2point_up_k_out, point2point_up_v_in, point2point_up_v_out,
                                                   point2point_up_num_heads,
                                                   point2point_q_in, point2point_q_out, point2point_k_in,
                                                   point2point_k_out, point2point_v_in,
                                                   point2point_v_out, point2point_num_heads, point2point_ff_conv1_in,
                                                   point2point_ff_conv1_out, point2point_ff_conv2_in,
                                                   point2point_ff_conv2_out)
            output_channels = point2point_ff_conv2_out[-1]
        if edgeconv_enable:
            self.block = EdgeConvBlock(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in,
                                       egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out,
                                       edgeconv_downsample_which, edgeconv_downsample_k, edgeconv_downsample_q_in,
                                       edgeconv_downsample_q_out,
                                       edgeconv_downsample_k_in, edgeconv_downsample_k_out, edgeconv_downsample_v_in,
                                       edgeconv_downsample_v_out, edgeconv_downsample_num_heads,
                                       edgeconv_upsample_which, edgeconv_upsample_q_in, edgeconv_upsample_q_out,
                                       edgeconv_upsample_k_in, edgeconv_upsample_k_out, edgeconv_upsample_v_in,
                                       edgeconv_upsample_v_out, edgeconv_upsample_num_heads,
                                       edgeconv_K, edgeconv_group_type, edgeconv_conv1_channel_in,
                                       edgeconv_conv1_channel_out,
                                       edgeconv_conv2_channel_in, edgeconv_conv2_channel_out)
            output_channels = edgeconv_conv2_channel_out[-1]

        self.conv = nn.Sequential(nn.Conv1d(output_channels, 1024, 1, bias=False), nn.BatchNorm1d(1024),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(output_channels + 2048 + 64, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Conv1d(128, 50, kernel_size=1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x, category_id):
        # x.shape == (B, 3, N)  category_id.shape == (B, 16, 1)
        B, C, N = x.shape
        # x.shape == (B, 3, N)
        x_tmp = self.block(x)
        # x_tmp.shape == (B, C, N)
        x = self.conv(x_tmp)
        # x.shape == (B, 1024, N)
        x_max = x.max(dim=-1, keepdim=True)[0]
        # x_max.shape == (B, 1024, 1)
        x_average = x.mean(dim=-1, keepdim=True)
        # x_average.shape == (B, 1024, 1)
        x = torch.cat([x_max, x_average], dim=1)
        # x.shape == (B, 2048, 1)
        category_id = self.conv1(category_id)
        # category_id.shape == (B, 64, 1)
        x = torch.cat([x, category_id], dim=1)
        # x.shape === (B, 2048+64, 1)
        x = x.repeat(1, 1, N)
        # x.shape == (B, 2048+64, N)
        x = torch.cat([x, x_tmp], dim=1)
        # x.shape == (B, 2048+64+C, N)
        x = self.conv2(x)
        # x.shape == (B, 256, N)
        x = self.dp1(x)
        # x.shape == (B, 256, N)
        x = self.conv3(x)
        # x.shape == (B, 256, N)
        x = self.dp2(x)
        # x.shape == (B, 256, N)
        x = self.conv4(x)
        # x.shape == (B, 128, N)
        x = self.conv5(x)
        # x.shape == (B, 50, N)
        return x


class SimplifiedEdgeConvBlock(nn.Module):
    def __init__(self, K=(32, 32), group_type=('center_diff', 'center_diff'),
                 conv1_channel_in=(3 * 2, 64 * 2), conv1_channel_out=(64, 64),
                 conv2_channel_in=(64, 64), conv2_channel_out=(64, 64)):
        super(SimplifiedEdgeConvBlock, self).__init__()

        self.edgeconv_list = nn.ModuleList([EdgeConv(k, g_type, conv1_in, conv1_out, conv2_in, conv2_out) for
                                            k, g_type, conv1_in, conv1_out, conv2_in, conv2_out in
                                            zip(K, group_type, conv1_channel_in, conv1_channel_out, conv2_channel_in,
                                                conv2_channel_out)])

    def forward(self, x):
        x_list = []
        for edgeconv in self.edgeconv_list:
            x = edgeconv(x)
            x_list.append(x)
        return x_list[-1]


class Global_CrossAttention(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_points=2048, num_heads=8):
        super(Global_CrossAttention, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        if q_out != v_out:
            raise ValueError('Please check the dimension of energy')
        print('q_out, k_out, v_out are same')
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
                 , neighbor_type='diff', len_feature=256, Att_Score_method='dot_product'):
        super(CrossAttention, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        if q_out != v_out:
            raise ValueError('Please check the dimension of energy')
        print('q_out, k_out, v_out are same')
        self.Att_Score_method = Att_Score_method
        self.neighbor_type = neighbor_type
        self.len_feature = len_feature
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
        self.p_cat = nn.Parameter(torch.ones(1, 2 * self.k_depth))  # q.shape == (1, 2D)

        self.two_c_type_linear = nn.Linear(2 * len_feature, len_feature)
        self.three_c_type_linear = nn.Linear(3 * len_feature, len_feature)
        self.dot_linear_energy = nn.Linear(3, num_heads)
        self.sub_linear_energy = nn.Linear(3, q_out)
        self.linear_q = nn.Linear(3, q_out)
        self.linear_k = nn.Linear(3, k_out)
        self.linear_v = nn.Linear(3, v_out)

    def forward(self, pcd, neighbors):
        # pcd.shape == (B, N, C)
        pcd = pcd.permute(0, 2, 1)[:, :, :, None]  # pcd.shape == (B, C, N, 1)
        if self.neighbor_type == 'center_neighbor' or 'center_diff' or 'neighbor_diff':
            neighbors = self.two_c_type_linear(neighbors)
        elif self.neighbor_type == 'center_neighbor_diff':
            neighbors = self.three_c_type_linear(neighbors)
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
        else:
            raise ValueError(f'Invalid value for Att_Score_method: {self.Att_Score_method}')

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x

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


#  multi_scale as separate keys
class CrossAttentionMS(nn.Module):
    def __init__(self, scale, shared_ca, concat_ms_inputs, mlp_or_ca, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64,
                 v_out=64, num_heads=8, ff_conv1_channels_in=64,
                 ff_conv1_channels_out=128, ff_conv2_channels_in=128, ff_conv2_channels_out=64):
        super(CrossAttentionMS, self).__init__()
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

    def forward(self, pcd, neighbor_list):
        if self.concat_ms_inputs:
            neighbors = torch.concat(neighbor_list, dim=1)
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
                    x_out = ca(pcd, neighbors)
                    # x_out.shape == (B, C, N)
                    x_output_list.append(x_out)
            if self.mlp_or_ca == 'mlp':
                x_out = torch.concat(x_output_list, dim=1)
                # x_out.shape == (B, C, N)
                x_out = self.linear(x_out)
                # x_out.shape == (B, C, N)
            else:
                neighbors = torch.stack(x_output_list, dim=-1)
                # x.shape == (B, C, N)   neighbors.shape == (B, C, N, K=scale+1)
                x_out = self.ca_aggregation(pcd, neighbors)
                # x_out.shape == (B, C, N)
        # x_out.shape == (B, C, N)
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
    def __init__(self, conv1_channel_in, conv1_channel_out,conv2_channel_in, conv2_channel_out):
        super(Point_Embedding, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(conv1_channel_in, conv1_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv1_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv2_channel_in, conv2_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv2_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, a, b, k):
        """
        :param a: a.shape == (B, N, C)
        :param b: b.shape == (B, M, C)
        :param k: int
        """
        a = a.permute(0, 2, 1)  # a.shape == (B, N, C)
        b = b.permute(0, 2, 1)  # b.shape == (B, M, C)
        idx = ops.knn(a, b, k)  # x_shape == (B, N, K)
        neighbors = ops.index_points(a, idx)[:, :, k, :]  # neighbors.shape == (B, N, K, C)
        diff = neighbors - a[:, :, None, :]  # diff.shape == (B, N, K, C)
        x = torch.cat([diff, a[:, :, k, :]], dim=3)  # x.shape == (B, N, K, 2C)
        x = self.conv1(x)  # x.shape == (B, N, K, out_features)
        x, _ = x.max(dim=2)  # x.shape == (B, N, out_features)
        idx = ops.knn(x, x, k)  # x_shape == (B, N, K)
        neighbors = ops.index_points(x, idx)[:, :, k, :]  # neighbors.shape == (B, N, K, C)
        diff = neighbors - x[:, :, None, :]  # diff.shape == (B, N, K, C)
        x = torch.cat([diff, x[:, :, k, :]], dim=3)  # x.shape == (B, N, K, 2C)
        x = self.conv2(x)  # x.shape == (B, N, K, out_features)
        x, _ = x.max(dim=2).permute(0, 2, 1)  # x.shape == (B, out_features, N)
        return x
