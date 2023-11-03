import torch
from torch import nn
import torch.nn.init as init

def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, K):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a ** 2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b ** 2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    # TODO: some values inside pairwise_distance is positive
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=K, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def fps(x, n_samples):
    """
    Farthest point sampling on pointcloud x
    :param x: pointcloud, tensor of shape (B, N, C)
    :param n_samples: number of points to sample
    :return: farthest point sampled pointcloud, tensor of shape (B, n_samples, C)
    """
    device = x.device
    B, N, C = x.shape

    centroids = torch.zeros(size=(B, n_samples), dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = x[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((x - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    output = x[batch_indices, centroids, :]

    return output


def extract_patches(x, centers, K):
    """
    Extract patches around centers
    :param x: pointcloud, tensor of shape (B, N, C)
    :param centroids: centroids, tensor of shape (B, n_samples)
    :param k: number of neighbors to consider
    :return: patches, list of tensors of shape (B, k, C)
    """
    B, n_samples = centers.shape
    N, C = x.shape[1:]

    batch_indices = torch.arange(B, dtype=torch.long).to(x.device)
    patches = []

    for i in range(n_samples):
        centroid = x[batch_indices, centers[:, i], :].view(B, 1, C)
        idx = knn(centroid, x, K)
        patch = index_points(x, idx)
        patches.append(patch)

    return patches


def select_neighbors(pcd, coordinate, K, neighbor_selection_method, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    coordinate = coordinate.permute(0, 2, 1)  # coordinate.shape == (B, N, C)

    if neighbor_selection_method == 'coordinate':
        idx = knn(coordinate, coordinate, K)  # idx.shape == (B, N, K)
    elif neighbor_selection_method == 'feature':
        idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    else:
        raise ValueError(
            f'neighbor_selection_method should be coordinate or feature, but got {neighbor_selection_method}')
    neighbors = index_points(pcd, idx)[:, :, ::K, :]  # neighbors.shape == (B, N, K, C)
    idx = idx[:, :, ::K]

    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'center_neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')

    return neighbors, idx


def group(pcd, coordinate, K, scale, neighbor_selection_method, group_type):
    if group_type == 'neighbor':
        neighbors, idx = select_neighbors(pcd, coordinate, K, scale, neighbor_selection_method,
                                          'neighbor')  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff, idx = select_neighbors(pcd, coordinate, K, scale, neighbor_selection_method,
                                     'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors, idx = select_neighbors(pcd, coordinate, K, scale, neighbor_selection_method, 'neighbor')
        # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff, idx = select_neighbors(pcd, coordinate, K, scale, neighbor_selection_method, 'diff')  # diff.shape == (
        # B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_neighbor_diff':
        neighbors, _ = select_neighbors(pcd, coordinate, K, scale, neighbor_selection_method,
                                        'neighbor')  # neighbors.shape == (B, C, N, K)
        diff, idx = select_neighbors(pcd, coordinate, K, scale, neighbor_selection_method,
                                     'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors, diff], dim=1)  # output.shape == (B,
        # 3C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor, center_diff or center_neighbor_diff, '
                         f'but got {group_type}')
    return output, idx  # or add coordinate info (+4)


def select_neighbors_single_scale(pcd, coordinate, K, scale, neighbor_selection_method, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    coordinate = coordinate.permute(0, 2, 1)  # coordinate.shape == (B, N, C)
    if neighbor_selection_method == 'coordinate':
        idx = knn(coordinate, coordinate, K * 2 ** scale)  # idx.shape == (B, N, K)
    elif neighbor_selection_method == 'feature':
        idx = knn(pcd, pcd, K * 2 ** scale)  # idx.shape == (B, N, K)
    else:
        raise ValueError(
            f'neighbor_selection_method should be coordinate or feature, but got {neighbor_selection_method}')

    start_idx = 0
    end_idx = K * 2 ** scale
    step = 2 ** scale
    neighbors = index_points(pcd, idx)[:, :, start_idx:end_idx:step, :]  # neighbor.shape == (B, N, K, C)
    idx_all = idx[:, :, start_idx:end_idx:step]

    if neighbor_type == 'neighbor':
        output = neighbors  # output.shape == (B, N, K, C)
    elif neighbor_type == 'diff':
        # pcd = index_points(pcd, idx_all)
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = diff  # output.shape == (B, N, K, C)
    elif neighbor_type == 'center_neighbor':
        output = torch.cat([pcd.unsqueeze(2).expand(-1, -1, K, -1), neighbors], dim=3)  # output.shape == (B, N, K, 2C)

    elif neighbor_type == 'center_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([pcd.unsqueeze(2).expand(-1, -1, K, -1), diff], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'neighbor_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([neighbors, neighbors - diff], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'center_neighbor_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([pcd.unsqueeze(2).expand(-1, -1, K, -1), neighbors, diff], dim=3)  # output.shape == (B, N, K, 3C)
    else:
        raise ValueError(f'group_type should be "neighbor", "diff", "center_neighbor", "center_diff", '
                         f'"neighbor_diff", "center_neighbor_diff" but got {neighbor_type}')

    return output, idx_all


#  in one key
def select_neighbors_in_one_key(pcd, coordinate, K, scale, neighbor_selection_method, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    coordinate = coordinate.permute(0, 2, 1)  # coordinate.shape == (B, N, C)

    if neighbor_selection_method == 'coordinate':
        idx = knn(coordinate, coordinate, K * (2 ** (scale + 1) - 1))  # idx.shape == (B, N, K)
    elif neighbor_selection_method == 'feature':
        idx = knn(pcd, pcd, K * (2 ** (scale + 1) - 1))  # idx.shape == (B, N, K)
    else:
        raise ValueError(
            f'neighbor_selection_method should be coordinate or feature, but got {neighbor_selection_method}')
    neighbors = []
    idx_all = []
    for i in range(scale + 1):
        start_idx = K * (2 ** i - 1)
        end_idx = K * (2 ** (i + 1) - 1)
        step = 2 ** i
        neighbor = index_points(pcd, idx)[:, :, start_idx:end_idx:step, :]  # neighbor.shape == (B, N, K, C)
        part_idx = idx[:, :, start_idx:end_idx:step]
        neighbors.append(neighbor)

        idx_all.append(part_idx)
    neighbors = torch.cat(neighbors, dim=2)  # neighbors.shape == (B, N, K, C)
    idx_all = torch.cat(idx_all, dim=2)

    if neighbor_type == 'neighbor':
        output = neighbors  # output.shape == (B, N, K, C)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = diff  # output.shape == (B, N, K, C)
    elif neighbor_type == 'center_neighbor':
        output = torch.cat([pcd[:, :, None, :], neighbors], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'center_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)

        output = torch.cat([pcd.unsqueeze(2).expand(-1, -1, K*(scale+1), -1), diff], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'neighbor_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([neighbors, neighbors - diff], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'center_neighbor_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([pcd.unsqueeze(2).expand(-1, -1, K*(scale+1), -1), neighbors, diff],
                           dim=3)  # output.shape == (B, N, K, 3C)
    else:
        raise ValueError(f'group_type should be "neighbor", "diff", "center_neighbor", "center_diff", '
                         f'"neighbor_diff", "center_neighbor_diff" but got {neighbor_type}')

    return output, idx_all # output.shape == (B, N, K, C) idx_all.shape == (B, N, K)


def select_neighbors_in_separate_key(pcd, coordinate, K, scale, neighbor_selection_method, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    coordinate = coordinate.permute(0, 2, 1)  # coordinate.shape == (B, N, C)

    if neighbor_selection_method == 'coordinate':
        idx = knn(coordinate, coordinate, K * (2 ** scale))  # idx.shape == (B, N, K)
    elif neighbor_selection_method == 'feature':
        idx = knn(pcd, pcd, K * (2 ** scale))  # idx.shape == (B, N, K)
    else:
        raise ValueError(
            f'neighbor_selection_method should be coordinate or feature, but got {neighbor_selection_method}')

    neighbor_list = []
    idx_all = []
    for i in range(scale + 1):
        start_idx = 0
        end_idx = K * (2 ** scale)
        step = 2 ** scale
        neighbors = index_points(pcd, idx)[:, :, start_idx:end_idx:step, :]  # neighbor.shape == (B, N, K, C)
        part_idx = idx[:, :, start_idx:end_idx:step]
        neighbor_list.append(neighbors)
        idx_all.append(part_idx)
    neighbors = torch.cat(neighbor_list, dim=2)  # neighbors.shape == (B, N, K, C)
    idx_all = torch.cat(idx_all, dim=2)

    if neighbor_type == 'neighbor':
        output = neighbors  # output.shape == (B, N, K, C)
    elif neighbor_type == 'diff':
        # pcd = index_points(pcd, idx_all)
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = diff  # output.shape == (B, N, K, C)
    elif neighbor_type == 'center_neighbor':
        output = torch.cat([pcd[:, :, None, :], neighbors], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'center_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([pcd.unsqueeze(2).expand(-1, -1, K*(scale+1), -1), diff], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'neighbor_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([neighbors, neighbors - diff], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'center_neighbor_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([pcd.unsqueeze(2).expand(-1, -1, K*(scale+1), -1), neighbors, diff],
                           dim=3)  # output.shape == (B, N, K, 3C)
    else:
        raise ValueError(f'group_type should be "neighbor", "diff", "center_neighbor", "center_diff", '
                         f'"neighbor_diff", "center_neighbor_diff" but got {neighbor_type}')

    return output, idx_all


def list_generator(neighbors, K, scale):
    sliced_tensors = []
    for i in range(scale + 1):
        start_idx = i * K
        end_idx = start_idx + K
        sliced_tensor = neighbors[:, :, :, start_idx:end_idx]
        sliced_tensors.append(sliced_tensor)
    return sliced_tensors


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Sequential(nn.Linear(1024, 512, bias=False),
                                     nn.BatchNorm1d(512),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                     nn.BatchNorm1d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = self.linear1(x)  # (batch_size, 1024) -> (batch_size, 512)
        x = self.dp1(x)
        x = self.linear2(x)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x