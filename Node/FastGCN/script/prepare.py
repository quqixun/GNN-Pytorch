"""dataset数据预处理

    1. 归一化节点特征
    2. 将节点划分为训练集、验证集和测试集
    3. 正则化邻接矩阵
    4. 加载数据至相应设备, cpu或gpu

"""


import torch
import scipy
import numpy as np

from .utils import PrepData, sparse_matrix_to_tensor


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
        dataset: PrepData, 包含的元素为:
                 X: tensor, 归一化后的节点特征
                 y: tensor, 节点类别标签
                 adjacency: sparse array, 正则化后的邻接矩阵
                 test_index: numpy array, 测试集样本索引
                 train_index: numpy array, 训练集样本索引
                 valid_index: numpy array, 验证集样本索引
                 adjacency_test: sparse array, 测试集节点邻接矩阵
                 adjacency_valid: sparse array, 验证集节点邻接矩阵
                 adjacency_train: sparse array, 训练集节点邻接矩阵

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

    # 正则化所有节点的邻接矩阵
    adjacency = dataset.data.adjacency.tocsr()
    adjacency = normalize_adjacency(adjacency)

    # 正则化训练集节点的邻接矩阵
    adjacency_train = adjacency[train_index, :][:, train_index]
    adjacency_train = normalize_adjacency(adjacency_train)

    # 获取验证集和测试集节点的邻接矩阵
    adjacency_test = adjacency[test_index, :]
    adjacency_valid = adjacency[valid_index, :]

    # 合并数据
    dataset = PrepData(
        X=X,
        y=y,
        adjacency=adjacency,
        test_index=test_index,
        train_index=train_index,
        valid_index=valid_index,
        adjacency_test=adjacency_test,
        adjacency_valid=adjacency_valid,
        adjacency_train=adjacency_train
    )

    return dataset
