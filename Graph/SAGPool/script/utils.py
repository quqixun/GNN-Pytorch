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


# 加载全局配置
def load_config(config_file):
    """加载全局配置

        加载模型参数和训练超参数, 用于不同的数据集训练模型

    """

    with open(config_file, 'r', encoding='utf-8') as f:
        # 读取yaml文件内容
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
