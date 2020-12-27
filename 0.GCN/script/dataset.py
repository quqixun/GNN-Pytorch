"""数据集的下载、存储与预处理

    原始数据集的下载、存储与预处理,
    原始数据集存储路径: $dataset_dir/$data/raw

"""


import os
import pickle
import numpy as np
import urllib.request

from .utils import *
from itertools import groupby
from scipy.sparse import coo_matrix


class Dataset(object):
    """数据集的下载、存储与预处理

        原始数据集的下载、存储与预处理,
        原始数据集存储路径: $dataset_dir/$data/raw

    """

    url_root = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    files = [
        'ind.{}.x', 'ind.{}.tx', 'ind.{}.allx',
        'ind.{}.y', 'ind.{}.ty', 'ind.{}.ally',
        'ind.{}.graph', 'ind.{}.test.index'
    ]

    def __init__(self, data, dataset_dir):
        """Dataset初始化

            原始数据集的下载、存储与预处理,
            原始数据集存储路径: $dataset_dir/$data/raw

            Inputs:
            -------
            data: string, 使用的数据集名称, ['cora', 'pubmed', 'citeseer']
            dataset_dir: string, 数据集保存根文件夹路径

        """

        assert data in ['cora', 'pubmed', 'citeseer'], 'uknown dataset'

        self.data = data
        self.data_dir = os.path.join(dataset_dir, data)

        # 预处理数据集不存在或更新现有数据集
        print('Downloading and Preprocessing [{}] Dataset ...'.format(data.upper()))
        self.__download_data()
        self.data = self.__process_data()

        return

    # ------------------------------------------------------------------------
    # 下载数据集

    def __download_data(self):
        """下载原始数据集

            分别下载各数据文件, 保存至self.data_dir文件夹

        """

        # 生成self.data_dir文件夹
        create_dir(self.data_dir)

        for name in self.files:
            # 遍历各数据文件
            name = name.format(self.data)
            file = os.path.join(self.data_dir, name)
            if not os.path.isfile(file):
                # 若数据文件不存在则下载文件
                url = '{}/{}'.format(self.url_root, name)
                self.__download_from_url(url, file)

        return

    def __download_from_url(self, url, file):
        """根据url下载文件

            从url下载文件保存至file

            Inputs:
            -------
            url: string, 文件下载链接
            file: string, 下载文件保存路径

        """

        try:
            # 建立文件连接, 写入文件
            data = urllib.request.urlopen(url, timeout=100)
            with open(file, 'wb') as f:
                f.write(data.read())
            data.close()
        except Exception:
            # 下载失败, 尝试再次下载
            self.__download_from_url(url, file)

        return

    # ------------------------------------------------------------------------
    # 预处理数据集

    def __process_data(self):
        """数据处理

            处理数据, 得到节点特征和标签, 邻接矩阵, 训练集, 验证集及测试集

            Output:
            -------
            dataset: Data tuple, 预处理后的数据集, 包含的元素为:
                     X: numpy array, 节点特征
                     y: numpy array, 节点类别标签
                     adjacency: sparse numpy array, 邻接矩阵
                     test_mask: numpy array, 测试集样本mask
                     train_mask: numpy array, 训练集样本mask
                     valid_mask: numpy array, 验证集样本mask

        """

        # 读取数据
        _, tx, allx, y, ty, ally, graph, test_index = \
            [self.read_data(os.path.join(self.data_dir, file.format(self.data))) for file in self.files]

        # 训练集与验证集样本索引
        train_index = np.arange(len(y))
        valid_index = np.arange(len(y), len(y) + 500)

        # 合并其他样本与测试集样本
        X = np.concatenate([allx, tx], axis=0)
        y = np.concatenate([ally, ty], axis=0).argmax(axis=1)

        # 根据图结构建立邻接矩阵
        adjacency = self.build_adjacency(graph)

        print(len(graph))

        if self.data == 'citeseer':
            # citeseer测试集中缺少若干节点

            # 从邻接矩阵中删除缺失节点的行和列
            # adjacency = adjacency.todense()
            full_test = list(range(min(test_index), adjacency.shape[0]))
            missing = [index for index in full_test if index not in test_index]
            print(missing)

            # 重新排序测试集索引
            new_test_index = list(range(min(test_index), min(test_index) + len(tx)))
            test_index_dict = {s: n for s, n in zip(sorted(test_index), new_test_index)}
            test_index = [test_index_dict[t] for t in test_index]

            new_graph = {}
            for key, values in graph.items():
                values = [v for v in values if v not in missing]
                if (key in missing) or (len(values) == 0):
                    continue

                key = test_index_dict[key]
                new_graph[key] = [test_index_dict[v] for v in values]
            print(new_graph)
            adjacency = self.build_adjacency(new_graph)

        # 测试集样本按顺序排列
        sorted_test_index = sorted(test_index)
        X[test_index] = X[sorted_test_index]
        y[test_index] = y[sorted_test_index]

        # 训练集, 验证集及测试集节点划分
        num_nodes = len(X)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask = np.zeros(num_nodes, dtype=bool)
        valid_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[test_index] = True
        train_mask[train_index] = True
        valid_mask[valid_index] = True

        # 合并数据
        dataset = Data(
            X=X,
            y=y,
            adjacency=adjacency,
            test_mask=test_mask,
            train_mask=train_mask,
            valid_mask=valid_mask
        )

        return dataset

    @staticmethod
    def read_data(file):
        """使用不同方式读取原始数据

            Input:
            ------
            file:, string, 需读取的文件路径

            Output:
            -------
            content: numpy array, 文件内容

        """

        file_name = os.path.basename(file)
        if 'test.index' in file_name:
            content = np.genfromtxt(file, dtype='int64')
        else:
            content = pickle.load(open(file, 'rb'), encoding='latin1')
            if hasattr(content, 'toarray'):
                content = content.toarray()

        return content

    @staticmethod
    def build_adjacency(graph):
        """根据图结构建立邻接矩阵

            Input:
            ------
            graph: dict, 每个节点的相邻节点字典

            Output:
            -------
            adjacency: numpy array, 邻接矩阵

        """

        # 每条边的两节点索引列表, 每一对节点表示一条边
        edge_index = []
        for src, dst in graph.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)

        # 删除重复的边
        sorted_edge_index = sorted(edge_index)
        edge_index = list(k for k, _ in groupby(sorted_edge_index))

        # 建立邻接矩阵
        num_nodes = len(graph)
        num_edges = len(edge_index)
        edge_index = np.asarray(edge_index)
        adjacency = coo_matrix((
            np.ones(num_edges),
            (edge_index[:, 0], edge_index[:, 1])
        ), shape=(num_nodes, num_nodes), dtype=float)

        return adjacency
