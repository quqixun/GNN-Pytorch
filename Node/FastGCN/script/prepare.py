"""dataset数据预处理

    1. 归一化节点特征
    2. 将节点划分为训练集、验证集和测试集
    3. 正则化邻接矩阵
    4. 加载数据至相应设备, cpu或gpu

"""


import torch
import scipy
import numpy as np

from .utils import Data


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

    adjacency += scipy.sparse.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = scipy.sparse.diags(np.power(degree, -0.5).flatten())
    norm_adjacency = d_hat.dot(adjacency).dot(d_hat)

    return norm_adjacency


def prepare(dataset):
    """数据预处理

        1. 归一化节点特征
        2. 将节点划分为训练集、验证集和测试集
        3. 正则化邻接矩阵
        4. 加载数据至相应设备, cpu或gpu

        Input:
        ------
        dataset: Data, 包含的元素为:
                 X: numpy array, 节点特征
                 y: numpy array, 节点类别标签
                 adjacency: sparse numpy array, 邻接矩阵
                 test_mask: numpy array, 测试集样本mask
                 train_mask: numpy array, 训练集样本mask
                 valid_mask: numpy array, 验证集样本mask

        Output:
        -------
        dataset: Data, 包含的元素为:
                 X: tensor, 归一化后的节点特征
                 y: tensor, 节点类别标签
                 adjacency: sparse tensor, 正则化后的邻接矩阵
                 test_mask: tensor, 测试集样本mask
                 train_mask: tensor, 训练集样本mask
                 valid_mask: tensor, 验证集样本mask

    """

    # 节点特征归一化
    X = dataset.data.X / dataset.data.X.sum(1, keepdims=True)
    X = torch.from_numpy(X)

    # 节点标签
    y = torch.from_numpy(dataset.data.y)

    # 各数据划分样本索引
    test_index = np.where(dataset.data.test_mask)[0]
    train_index = np.where(dataset.data.train_mask)[0]
    valid_index = np.where(dataset.data.valid_mask)[0]

    adjacency = dataset.data.adjacency.tocsr()
    adjacenty_train = adjacency[train_index, :][:, train_index]

    adjacency = normalize_adjacency(adjacency)
    adjacenty_test = adjacency[test_index, :]
    adjacenty_valid = adjacency[valid_index, :]
    adjacenty_train = normalize_adjacency(adjacenty_train)

    # # 邻接矩阵正则化
    # norm_adjacency = normalize_adjacency(dataset.data.adjacency)
    # indices = np.asarray([norm_adjacency.row, norm_adjacency.col])
    # indices = torch.from_numpy(indices.astype(int)).long()
    # values = torch.from_numpy(norm_adjacency.data.astype(np.float32))
    # adjacency = torch.sparse.FloatTensor(indices, values, (len(X), len(X)))

    # # 数据加载至的设备
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # 合并数据
    # dataset = Data(
    #     X=X.to(device),
    #     y=y.to(device),
    #     adjacency=adjacency.to(device),
    #     test_index=test_index.to(device),
    #     train_index=train_index.to(device),
    #     valid_index=valid_index.to(device)
    # )

    # return dataset

    return None
