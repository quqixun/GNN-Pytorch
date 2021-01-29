"""辅助函数
"""


import os
import yaml
import scipy
import torch
import numpy as np


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


def normalize_adjacency(adjacency, batch):
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

    device = adjacency.device
    adjacency = adjacency.detach().cpu().numpy()
    batch = np.logical_not(batch.detach().cpu().numpy())

    # 正则化邻接矩阵
    adjacency += np.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = np.diag(np.power(degree, -0.5).flatten())
    adjacency = d_hat.dot(adjacency).dot(d_hat)

    adjacency[batch, :] = 0
    adjacency[:, batch] = 0

    adjacency_tensor = torch.FloatTensor(adjacency)
    adjacency_tensor = adjacency_tensor.to(device)

    return adjacency_tensor


def batch_normalize_adjacency(adjacency, batch):
    """对batch中每张图的邻接矩阵做正则化
    """

    device = adjacency.device

    norm_adjacency = []
    for adj, b in zip(adjacency, batch):
        norm_adj = normalize_adjacency(adj, b)
        norm_adj = norm_adj.unsqueeze(0)
        norm_adjacency.append(norm_adj)
    norm_adjacency = torch.cat(norm_adjacency)
    norm_adjacency = norm_adjacency.to(device)

    return norm_adjacency
