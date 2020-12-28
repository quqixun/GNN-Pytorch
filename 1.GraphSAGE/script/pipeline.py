"""GraphSAGE模型训练与预测
"""


import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .model import GraphSAGE
from .sampling import multihop_sampling


class Pipeline(object):
    """GraphSAGE模型训练与预测
    """

    def __init__(self, **params):
        """GraphSAGE模型训练与预测

            加载GCN模型, 生成训练必要组件实例

            Input:
            ------
            params: dict, 模型参数和超参数, 格式为:
                    {
                        'model': {
                            'input_dim': 1433,              # 节点特征维度
                            'hidden_dims': [128, 7],        # 隐层输出特征维度
                            'num_neighbors_list': [10, 10]  # 没接采样邻居的节点数
                        },
                        'hyper': {
                            'lr': 3e-3,                # 优化器初始学习率
                            'epochs': 10,              # 训练轮次
                            'batch_size': 16,          # 批数据大小
                            'weight_decay': 5e-4,      # 优化器权重衰减
                            'num_batch_per_epoch': 20  # 每个epoch循环的批次数
                        }
                    }

        """

        # 获取可用的计算设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.__init_environment(params['random_state'])
        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])

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

        # 建立模型
        self.model = GraphSAGE(**model_params)
        self.model = self.model.to(self.device)

        # 每一阶采样的邻居数量列表
        self.num_neighbors_list = model_params['num_neighbors_list']

        return

    def __build_components(self, **hyper_params):
        """加载组件

            Input:
            ------
            hyper_params: dict, 超参数

        """

        # 训练过程参数
        self.epochs = hyper_params['epochs']
        self.batch_size = hyper_params['batch_size']
        self.num_batch_per_epoch = hyper_params['num_batch_per_epoch']

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 定义优化器
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['weight_decay']
        )

        return

    def train(self, dataset):
        """训练模型

            Input:
            ------
            dataset: Data, 包含X, y, adjacency_dict, test_index,
                     train_index和valid_index

        """

        # 记录验证集效果最佳模型
        best_model = None

        # 记录最佳的验证集准确率
        best_valid_acc = 0

        for epoch in range(self.epochs):
            # 模型训练模式
            self.model.train()

            # 用于记录每个epoch中所有batch的loss
            epoch_losses = []

            for batch in range(self.num_batch_per_epoch):
                # 在每一个batch中, 对节点进行采样, 并对节点的多阶邻居进行采样
                train_index = np.random.choice(dataset.train_index, size=(self.batch_size,))
                neighbor_index = multihop_sampling(train_index, self.num_neighbors_list, dataset.adjacency_dict)

                # 采样节点的标签和其邻居(包含采样节点本身)的特征
                train_y = torch.from_numpy(dataset.y[train_index]).long().to(self.device)
                neighbor_X = [torch.from_numpy(dataset.X[index]).float().to(self.device) for index in neighbor_index]

                # 模型输出
                train_logits = self.model(neighbor_X)

                # 计算损失函数
                loss = self.criterion(train_logits, train_y)
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
            dataset: Data, 包含X, y, adjacency_dict, test_index,
                     train_index和valid_index
            split: string, 待预测的节点

            Output:
            -------
            accuracy: float, 节点分类准确率

        """

        # 模型推断模式
        self.model.eval()

        # 数据划分节点索引
        if split == 'train':
            index = dataset.train_index
        elif split == 'valid':
            index = dataset.valid_index
        else:  # split == 'test'
            index = dataset.test_index

        # 数据划分节点的标签和其邻居(包含节点本身)的特征
        neighbor_index = multihop_sampling(index, self.num_neighbors_list, dataset.adjacency_dict)
        neighbor_X = [torch.from_numpy(dataset.X[index]).float().to(self.device) for index in neighbor_index]

        # 获得待预测节点的输出
        logits = self.model(neighbor_X)
        predict_y = logits.max(1)[1]

        # 计算预测准确率
        y = torch.from_numpy(dataset.y[index]).long().to(self.device)
        accuracy = torch.eq(predict_y, y).float().mean()

        return accuracy
