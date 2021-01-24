"""GAT模型训练与预测
"""


import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .model import GAT


class Pipeline(object):
    """GAT模型训练与预测
    """

    def __init__(self, **params):
        """GAT模型训练与预测

            加载GAT模型, 生成训练必要组件实例

            Input:
            ------
            params: dict, 模型参数和超参数, 格式为:
                    {
                        'sparse': False,
                        'random_state' 42,
                        'model': {
                            'input_dim': 1433,
                            'hidden_dim': 8,
                            'output_dim': 7,
                            'num_heads': 8,
                            'dropout': 0.6,
                            'alpha': 0.2
                        },
                        'hyper': {
                            'lr': 3e-3,
                            'epochs': 10,
                            'patience': 100,
                            'weight_decay': 5e-4
                        }
                    }

        """

        self.sparse = params['sparse']
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

        self.model = GAT(sparse=self.sparse, **model_params)
        if torch.cuda.is_available():
            self.model.cuda()

        return

    def __build_components(self, **hyper_params):
        """加载组件

            Input:
            ------
            hyper_params: dict, 超参数

        """

        self.epochs = hyper_params['epochs']
        self.patience = hyper_params['patience']

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.NLLLoss()

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
            dataset: Data, 包含X, y, adjacency, test_index,
                     train_index和valid_index

        """

        # 训练集标签
        train_y = dataset.y[dataset.train_index]

        best_model = None

        # 记录最佳的验证集准确率
        best_valid_acc = 0

        # 获得最佳的验证集后计数轮次
        epochs_after_best = 0

        for epoch in range(self.epochs):
            # 模型训练模式
            self.model.train()

            # 模型输出
            logits = self.model(dataset.X, dataset.edges)
            train_logits = logits[dataset.train_index]

            # 计算损失函数
            loss = self.criterion(train_logits, train_y)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 计算训练集准确率
            train_acc = self.predict(dataset, 'train')
            # 计算验证集准确率
            valid_acc = self.predict(dataset, 'valid')

            print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]-[ValidAcc:{:.4f}]'.format(
                epoch, loss, train_acc, valid_acc))

            if valid_acc >= best_valid_acc:
                best_model = copy.deepcopy(self.model)
                # 获得最佳验证集准确率
                best_valid_acc = valid_acc
                # 从新计数轮次
                epochs_after_best = 0
            else:
                # 未获得最佳验证集准确率
                # 增加计数轮次
                epochs_after_best += 1

            if epochs_after_best == self.patience:
                # 符合早停条件
                self.model = best_model
                break

        return

    def predict(self, dataset, split='train'):
        """模型预测

            Inputs:
            -------
            dataset: Data, Data, 包含X, y, adjacency, test_index,
                     train_index和valid_index
            split: string, 待预测的节点

            Output:
            -------
            accuracy: float, 节点分类准确率

        """

        # 模型推断模式
        self.model.eval()

        # 节点索引
        if split == 'train':
            index = dataset.train_index
        elif split == 'valid':
            index = dataset.valid_index
        else:  # split == 'test'
            index = dataset.test_index

        # 获得待预测节点的输出
        logits = self.model(dataset.X, dataset.edges)
        predict_y = logits[index].max(1)[1]

        # 计算预测准确率
        y = dataset.y[index]
        accuracy = torch.eq(predict_y, y).float().mean()

        return accuracy
