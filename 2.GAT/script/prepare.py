"""Cora数据预处理

    1. 归一化节点特征
    2. 将节点划分为训练集、验证集和测试集
    3. 正则化邻接矩阵
    4. 加载数据至相应设备, cpu或gpu

"""


import torch
import scipy
import numpy as np

from .utils import *


def normalization(adjacency):
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

    adjacency += scipy.sparse.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = scipy.sparse.diags(np.power(degree, -0.5).flatten())
    norm_adjacency = d_hat.dot(adjacency).dot(d_hat).tocoo()

    return norm_adjacency


def prepare_data(cora, sparse=False):
    """Cora数据预处理

        1. 归一化节点特征
        2. 将节点划分为训练集、验证集和测试集
        3. 正则化邻接矩阵
        4. 加载数据至相应设备, cpu或gpu

        Input:
        ------
        cora: Data, 包含的元素为:
              X: numpy array, 节点特征
              y: numpy array, 节点类别标签
              adjacency: sparse numpy array, 邻接矩阵
              test_mask: numpy array, 测试集样本mask
              train_mask: numpy array, 训练集样本mask
              valid_mask: numpy array, 验证集样本mask
        sparse: boolean, 是否使用稀疏矩阵

        Output:
        -------
        dataset: Data, 包含的元素为:
                 X: tensor, 归一化后的节点特征
                 y: tensor, 节点类别标签
                 adjacency: tensor, 正则化后的邻接矩阵
                 test_index: tensor, 测试集样本索引
                 train_index: tensor, 训练集样本索引
                 valid_index: tensor, 验证集样本索引

    """

    # 节点特征归一化
    X = cora.data.X / cora.data.X.sum(1, keepdims=True)
    X = torch.FloatTensor(X)

    # 节点标签
    y = torch.LongTensor(cora.data.y)

    # 各数据划分样本索引
    test_index = torch.LongTensor(np.where(cora.data.test_mask)[0])
    train_index = torch.LongTensor(np.where(cora.data.train_mask)[0])
    valid_index = torch.LongTensor(np.where(cora.data.valid_mask)[0])

    # 邻接矩阵正则化
    norm_adjacency = normalization(cora.data.adjacency)
    indices = np.asarray([norm_adjacency.row, norm_adjacency.col])
    indices = torch.from_numpy(indices.astype(int)).long()
    values = torch.from_numpy(norm_adjacency.data.astype(np.float32))
    adjacency = torch.sparse.FloatTensor(indices, values, (2708, 2708))

    if not sparse:
        # 不使用稀疏矩阵
        adjacency = adjacency.to_dense()

    # 数据加载至的设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 合并数据
    dataset = PrepData(
        X=X.to(device),
        y=y.to(device),
        adjacency=adjacency.to(device),
        test_index=test_index.to(device),
        train_index=train_index.to(device),
        valid_index=valid_index.to(device)
    )

    return dataset
