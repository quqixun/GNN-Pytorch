"""辅助函数
"""


import os
import yaml
import torch
import numpy as np

from collections import namedtuple
from scipy.sparse.csr import csr_matrix
from scipy.sparse.coo import coo_matrix


def create_dir(dir_path):
    # 生成文件夹
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


# 定义数据结构
Data = namedtuple(
    typename='Data',
    field_names=[
        'X',           # 节点特征
        'y',           # 节点类别标签
        'adjacency',   # 邻接矩阵
        'test_mask',   # 测试集样本mask
        'train_mask',  # 训练集样本mask
        'valid_mask'   # 验证集样本mask
    ]
)


# 定义预处理后的数据结构
PrepData = namedtuple(
    typename='PrepData',
    field_names=[
        'X',                # 节点特征
        'y',                # 节点类别标签
        'adjacency',        # 邻接矩阵
        'test_index',       # 测试集样本索引
        'train_index',      # 训练集样本索引
        'valid_index',      # 验证集样本索引
        'adjacency_train'   # 训练集节点邻接矩阵
    ]
)


# 加载全局配置
def load_config(config_file):
    """加载全局配置

        加载模型参数和训练超参数, 用于不同的数据集训练模型

    """

    with open(config_file, 'r', encoding='utf-8') as f:
        # 读取yaml文件内容
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def sparse_matrix_to_tensor(sparse_matrix, device):
    """稀疏矩阵转换至tensor

        Input:
        ------
        sparse_matrix: csr_matrix or coo_matrix, 输入稀疏矩阵

        Output:
        -------
        sparse_tensor: tensor, 稀疏tensor

    """

    # 检查输入稀疏矩阵类型, 后续使用coo_matrix做处理
    assert isinstance(sparse_matrix, (csr_matrix, coo_matrix))
    if isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocoo()

    # 转换成稀疏tensor
    indices = np.asarray([sparse_matrix.row, sparse_matrix.col])
    indices = torch.from_numpy(indices.astype(int)).long()
    values = torch.from_numpy(sparse_matrix.data.astype(np.float32))
    sparse_tensor = torch.sparse.FloatTensor(indices, values, sparse_matrix.shape)

    sparse_tensor = sparse_tensor.to(device)
    return sparse_tensor
