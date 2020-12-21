"""辅助函数
"""


import os

from collections import namedtuple


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
