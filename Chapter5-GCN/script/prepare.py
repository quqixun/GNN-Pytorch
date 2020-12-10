"""数据准备
"""


import torch
import scipy
import numpy as np

from .utils import *
from collections import namedtuple


Data = namedtuple(
    typename='Data',
    field_names=[
        'X',
        'y',
        'adjacency',
        'test_mask',
        'train_mask',
        'valid_mask'
    ]
)


def normalization(adjacency):
    """
    """

    adjacency += scipy.sparse.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = scipy.sparse.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()

    return L


def prepare_data(cora):
    """
    """

    print(TITLE_STRING.format('DATASET PREPARETION'))

    # 节点特征归一化
    X = cora.data.X / cora.data.X.sum(1, keepdims=True)
    X = torch.from_numpy(X)

    # 节点标签
    y = torch.from_numpy(cora.data.y)

    # 节点划分
    test_mask = torch.from_numpy(cora.data.test_mask)
    train_mask = torch.from_numpy(cora.data.train_mask)
    valid_mask = torch.from_numpy(cora.data.valid_mask)

    # 邻接矩阵正则化
    norm_adjacency = normalization(cora.data.adjacency)
    indices = np.asarray([norm_adjacency.row, norm_adjacency.col])
    indices = torch.from_numpy(indices.astype(int)).long()
    values = torch.from_numpy(norm_adjacency.data.astype(float))
    adjacency = torch.sparse.FloatTensor(indices, values, (2708, 2708))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Data(
        X=X.to(device),
        y=y.to(device),
        adjacency=adjacency.to(device),
        test_mask=test_mask.to(device),
        train_mask=train_mask.to(device),
        valid_mask=valid_mask.to(device)
    )

    return dataset
