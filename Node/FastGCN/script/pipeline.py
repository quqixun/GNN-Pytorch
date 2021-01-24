"""FastGCN训练与预测
"""


import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .model import GCN
from .sampling import Sampler
from .utils import sparse_matrix_to_tensor


class Pipeline(object):
    """FastGCN训练与预测
    """

    def __init__(self, **params):
        """FastGCN训练与预测

            加载FastGCN模型, 生成训练必要组件实例

            Input:
            ------
            params: dict, 模型参数和超参数, 格式为:
                    {
                        'random_state': 42,
                        'model': {
                            'input_dim': 1433,
                            'output_dim': 7,
                            'hidden_dim': 16,
                            'use_bias': True,
                            'dropout': 0.5
                        },
                        'hyper': {
                            'lr': 1e-2,
                            'epochs': 100,
                            'batch_size': 64,
                            'weight_decay': 5e-4,
                            'sampler_dims': [128, 128]
                        }
                    }

        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__init_environment(params['random_state'])
        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])
        self.__build_sampler(**params)

        return

    def __init_environment(self, random_state):
        """初始化环境

            Input:
            ------
            random_state: int, 随机种子

        """

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return

    def __build_model(self, **model_params):
        """加载模型

            Input:
            ------
            model_params: dict, 模型相关参数

        """

        self.model = GCN(**model_params)
        self.model.to(self.device)

        return

    def __build_components(self, **hyper_params):
        """加载组件

            Input:
            ------
            hyper_params: dict, 超参数

        """

        self.epochs = hyper_params['epochs']
        self.batch_size = hyper_params['batch_size']

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 定义优化器
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['weight_decay']
        )

        return

    def __build_sampler(self, **params):
        """加载节点采样器

            Input:
            ------
            params: dict, 模型参数和超参数, 提供的参数包括:
                    input_dim: 节点特征维度
                    sampler_dims: 每层网络采样节点数

        """

        self.sampler = Sampler(
            input_dim=params['model']['input_dim'],
            sampler_dims=params['hyper']['sampler_dims'],
            device=self.device
        )

        return

    def __batch_generator(self, nodes, y, shuffle=True):
        """批数据生成器

            Inputs:
            -------
            nodes: numpy array, 用于生成批数据的节点索引列表
            y: tensor, 节点标签
            shuffle: boolean, 是否打乱数据再生成批数据

            Output:
            -------
            batch_nodes: numpy array, 批数据节点索引列表
            batch_y: tensor, 批数据标签

        """

        if shuffle:
            # 打乱输入的节点索引列表
            np.random.shuffle(nodes)

        start = 0
        while True:
            # 获得当前批数据终点索引
            end = start + self.batch_size
            if end > len(nodes):
                break

            # 获得批数据节点索引列表
            batch_nodes = nodes[start:end]
            # 获得批数据标签
            batch_y = y[batch_nodes]
            yield batch_nodes, batch_y

            # 计算下一批数据的起点索引
            start += self.batch_size

        return

    def train(self, dataset):
        """训练模型

            Input:
            ------
            dataset: Data, 包含X, y, adjacency, test_index,
                     train_index和valid_index

        """

        # 训练集标签
        train_nodes = dataset.train_index
        train_X = dataset.X[train_nodes]
        train_y = dataset.y[train_nodes]

        # 记录验证集效果最佳模型
        best_model = None

        # 记录验证集最佳准确率
        best_valid_acc = 0

        for epoch in range(self.epochs):
            # 模型训练模式
            self.model.train()

            # 用于记录每个epoch中所有batch的loss
            epoch_losses = []

            for batch_nodes, batch_y in self.__batch_generator(dataset.train_index, train_y):

                # 获得采样节点特征及采样的邻接矩阵
                sampled_X, sampled_adjacency = self.sampler.sampling(
                    X=train_X,
                    adjacency=dataset.adjacency_train,
                    batch_nodes=batch_nodes
                )

                # 模型输出
                logits = self.model(sampled_adjacency, sampled_X)

                # 计算损失函数
                loss = self.criterion(logits, batch_y)
                epoch_losses.append(loss.item())

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 计算epoch中所有batch的loss的均值
            epoch_loss = np.mean(epoch_losses)

            # 计算训练集准确率
            train_acc = self.predict(dataset, 'train')
            # 计算验证集准确率
            valid_acc = self.predict(dataset, 'valid')

            print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]-[ValidAcc:{:.4f}]'.format(
                epoch, epoch_loss, train_acc, valid_acc))

            if valid_acc >= best_valid_acc:
                # 获得最佳验证集准确率
                best_model = copy.deepcopy(self.model)
                best_valid_acc = valid_acc

        # 最终模型为验证集效果最佳的模型
        self.model = best_model

        return

    def predict(self, dataset, split='train'):
        """模型预测

            Inputs:
            -------
            dataset: Data, 包含X, y, adjacency, test_index,
                     train_index和valid_index
            split: string, 待预测的节点

            Output:
            -------
            accuracy: float, 节点分类准确率

        """

        # 模型推断模式
        self.model.eval()

        # 节点mask
        if split == 'train':
            index = dataset.train_index
        elif split == 'valid':
            index = dataset.valid_index
        else:  # split == 'test'
            index = dataset.test_index

        # 数据集对应的邻接矩阵
        split_adjacency = dataset.adjacency[index, :]
        split_adjacency = sparse_matrix_to_tensor(split_adjacency, self.device)

        # 完整邻接矩阵
        adjacency = sparse_matrix_to_tensor(dataset.adjacency, self.device)

        # 获得待预测节点的输出
        logits = self.model([adjacency, split_adjacency], dataset.X)
        predict_y = logits.max(1)[1]

        # 计算预测准确率
        y = dataset.y[index]
        accuracy = torch.eq(predict_y, y).float().mean()
        return accuracy
