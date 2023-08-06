import torch


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


def     knn(a, b, k):
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
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
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


def extract_patches(x, centers, k):
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
        idx = knn(centroid, x, k)
        patch = index_points(x, idx)
        patches.append(patch)

    return patches


def select_neighbors(pcd, coordinate, K, scale, neighbor_selection_method, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    coordinate = coordinate.permute(0, 2, 1)  # coordinate.shape == (B, N, C)

    K = K * 2 ** scale
    if neighbor_selection_method == 'coordinate':
        idx = knn(coordinate, coordinate, K)  # idx.shape == (B, N, K)
    elif neighbor_selection_method == 'feature':
        idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    else:
        raise ValueError(
            f'neighbor_selection_method should be coordinate or feature, but got {neighbor_selection_method}')
    neighbors = index_points(pcd, idx)[:, :, ::2 ** scale, :]  # neighbors.shape == (B, N, K, C)
    idx = idx[:, :, ::2 ** scale]

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


#  in one key
def select_neighbors_in_one_key(pcd, coordinate, K, scale, neighbor_selection_method, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    coordinate = coordinate.permute(0, 2, 1)  # coordinate.shape == (B, N, C)

    # K = K * (2 ** scale - 1)  # total number of neighbors

    if neighbor_selection_method == 'coordinate':
        idx = knn(coordinate, coordinate, K)  # idx.shape == (B, N, K)
    elif neighbor_selection_method == 'feature':
        idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    else:
        raise ValueError(
            f'neighbor_selection_method should be coordinate or feature, but got {neighbor_selection_method}')
    neighbors = []
    idx_all = []
    for i in range(scale+1):
        neighbor = index_points(pcd, idx)[:, :, K * (2 ** scale - 1):K * (2 ** (scale + 1) - 1):2 ** scale, :]
        # neighbor.shape == (B, N, K, C)
        idx = idx[:, :, K * (2 ** scale - 1):K * (2 ** (scale + 1) - 1):2 ** scale]
        neighbors.append(neighbor)
        idx_all.append(idx)
    neighbors = torch.cat(neighbors, dim=2)  # neighbors.shape == (B, N, K, C)
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
        output = torch.cat([pcd[:, :, None, :], diff], dim=3)  # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'neighbor_diff':
        output = torch.cat([neighbors, neighbors - pcd[:, :, None, :]])
        # output.shape == (B, N, K, 2C)
    elif neighbor_type == 'center_neighbor_diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        output = torch.cat([pcd[:, :, None, :], neighbors, diff], dim=3)
        # output.shape == (B, N, K, 3C)
    else:
        raise ValueError(f'group_type should be "neighbor", "diff", "center_neighbor", "center_diff", '
                         f'"neighbor_diff", "center_neighbor_diff" but got {neighbor_type}')

    return output, idx_all


def operation_mode(operation_mode, query, keys):
    if operation_mode == 'multiple':
        attention_maps = [torch.matmul(query.unsqueeze(-3), key.transpose(-1, -2)).softmax(dim=-1) for key in keys]
    elif operation_mode == 'addition':
        attention_maps = [(query.unsqueeze(-3) + key.transpose(-1, -2)).softmax(dim=-1) for key in keys]
    elif operation_mode == 'subtraction':
        attention_maps = [(query.unsqueeze(-3) - key.transpose(-1, -2)).softmax(dim=-1) for key in keys]
    else:
        raise ValueError(f'Unsupported operation mode: {operation_mode}. It should be multiple, addition or '
                         f'subtraction.')


