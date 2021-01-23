"""辅助函数
"""


import os
import yaml
import scipy
import torch
import numpy as np

from itertools import groupby
from scipy.sparse.coo import coo_matrix


def create_dir(dir_path):
    # 生成文件夹
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


def load_config(config_file):
    """加载全局配置

        加载模型参数和训练超参数, 用于不同的数据集训练模型

    """

    with open(config_file, 'r', encoding='utf-8') as f:
        # 读取yaml文件内容
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def generate_adjacency(X, graph):
    """根据边列表生成邻接矩阵
    """

    num_nodes = X.size(0)
    num_edges = graph.size(1)
    graph_np = graph.detach().cpu().numpy()

    edge_index = np.concatenate((graph_np, np.flipud(graph_np)), axis=1)
    edge_index = edge_index.T.tolist()
    sorted_edge_index = sorted(edge_index)
    edge_index = list(k for k, _ in groupby(sorted_edge_index))
    edge_index = np.asarray(edge_index)

    # 生成稀疏邻接矩阵
    adjacency = coo_matrix((
        np.ones(num_edges),
        (edge_index[:, 0], edge_index[:, 1])
    ), shape=(num_nodes, num_nodes), dtype=float)

    return adjacency


def normalize_adjacency(adjacency):
    """邻接矩阵正则化

        L = D^-0.5 * (A + I) * D^-0.5
        A: 邻接矩阵, L: 正则化邻接矩阵

        Input:
        ------
        adjacency: sparse numpy array, 邻接矩阵

        Output:
        -------
        norm_adjacency: sparse numpy array, 正则化邻接矩阵

    """

    # 正则化邻接矩阵
    adjacency += scipy.sparse.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = scipy.sparse.diags(np.power(degree, -0.5).flatten())
    adjacency = d_hat.dot(adjacency).dot(d_hat).tocoo()

    # 转换成稀疏tensor
    indices = np.asarray([adjacency.row, adjacency.col])
    indices = torch.from_numpy(indices.astype(int)).long()
    values = torch.from_numpy(adjacency.data.astype(np.float32))
    adjacency_tensor = torch.sparse.FloatTensor(indices, values, adjacency.shape)

    return adjacency_tensor


def topk(node_score, graph_batch, keep_ratio):
    """获取每个图的topk个重要的节点
    """

    graph_ids = list(set(graph_batch.cpu().numpy()))
    mask = node_score.new_empty((0,), dtype=torch.bool)

    for grid_id in graph_ids:
        graph_node_score = node_score[graph_batch == grid_id].view(-1)
        _, sorted_index = graph_node_score.sort(descending=True)

        num_graph_node = len(graph_node_score)
        graph_mask = node_score.new_zeros((num_graph_node,), dtype=torch.bool)

        num_keep_node = int(keep_ratio * num_graph_node)
        num_keep_node = max(num_keep_node, 1)
        graph_mask[sorted_index[:num_keep_node]] = True
        mask = torch.cat([mask, graph_mask])

    return mask


def filter_adjacency(adjacency, mask):
    """生成仅包含重要节点的邻接矩阵
    """

    device = adjacency.device
    num_nodes = adjacency.size(0)

    indices = adjacency.coalesce().indices().cpu().numpy()
    row, col = indices

    non_self_loop = row != col
    row = row[non_self_loop]
    col = col[non_self_loop]

    sparse_adjacency = scipy.sparse.csr_matrix(
        (np.ones(len(row)), (row, col)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )

    mask = mask.cpu().numpy()
    filtered_adjacency = sparse_adjacency[mask, :][:, mask]
    filtered_adjacency = normalize_adjacency(filtered_adjacency)
    filtered_adjacency = filtered_adjacency.to(device)

    return filtered_adjacency
