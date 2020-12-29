"""数据预处理

    1. 归一化节点特征
    2. 将节点划分为训练集、验证集和测试集
    3. 正则化邻接矩阵
    4. 加载数据至相应设备, cpu或gpu

"""


import numpy as np

from .utils import PrepData


def prepare(dataset):
    """数据预处理

        1. 归一化节点特征
        2. 将节点划分为训练集、验证集和测试集

        Input:
        ------
        dataset: Data, 包含的元素为:
                 X: numpy array, 节点特征
                 y: numpy array, 节点类别标签
                 test_mask: numpy array, 测试集样本mask
                 train_mask: numpy array, 训练集样本mask
                 valid_mask: numpy array, 验证集样本mask
                 adjacency_dict: dict, 节点邻居字典

        Output:
        -------
        dataset: PrepData, 包含的元素为:
                 X: tensor, 归一化后的节点特征
                 y: tensor, 节点类别标签
                 adjacency: sparse tensor, 正则化后的邻接矩阵
                 test_mask: tensor, 测试集样本mask
                 train_mask: tensor, 训练集样本mask
                 valid_mask: tensor, 验证集样本mask

    """

    # 节点特征归一化
    X = dataset.data.X / dataset.data.X.sum(1, keepdims=True)

    # 各数据划分样本索引
    test_index = np.where(dataset.data.test_mask)[0]
    train_index = np.where(dataset.data.train_mask)[0]
    valid_index = np.where(dataset.data.valid_mask)[0]

    # 合并数据
    dataset = PrepData(
        X=X,
        y=dataset.data.y,
        test_index=test_index,
        train_index=train_index,
        valid_index=valid_index,
        adjacency_dict=dataset.data.adjacency_dict
    )

    return dataset
