"""GRAND模型训练与预测
"""


import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .grand import *
from .model import MLP


class Pipeline(object):
    """GRAND模型训练与预测
    """

    def __init__(self, **params):
        """GRAND模型训练与预测

            加载GCN模型, 生成训练必要组件实例

            Input:
            ------
            params: dict, 模型参数和超参数, 格式为:
                    {
                        'random_state': 42,
                        'grand': {
                            'S': 4,
                            'K': 5,
                            'T': 0.5,
                            'L': 1.0,
                            'D': 0.5
                        },
                        'model': {
                            'input_dim': 1433,
                            'output_dim': 7,
                            'hidden_dim': 32,
                            'dropout': 0.5
                            'use_bn': False
                        },
                        'hyper': {
                            'lr': 1e-2,
                            'epochs': 1000,
                            'patience': 100,
                            'weight_decay': 5e-4
                        }
                    }

        """

        self.__init_environment(params['random_state'])
        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])
        self.GRAND = GRAND(**params['grand'])

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

        self.model = MLP(**model_params)
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
            dataset: Data, 包含X, y, adjacency, test_mask,
                     train_mask和valid_mask

        """

        # 训练集标签
        train_y = dataset.y[dataset.train_index]

        # 记录验证集效果最佳模型
        best_model = None

        # 记录验证集最佳准确率
        best_valid_acc = 0

        # 获得最佳的验证集后计数轮次
        epochs_after_best = 0

        for epoch in range(self.epochs):
            # 模型训练模式
            self.model.train()

            # Augmentation过程, 包含DropNode合多阶聚合过程
            Xs = self.GRAND.random_propagate(
                adjacency=dataset.adjacency,
                X=dataset.X, train=True
            )

            # 对每组Augmentation的数据分别做预测并计算分类损失函数
            outputs, cls_loss = [], 0
            for X in Xs:
                logits = self.model(X)[dataset.train_index]
                outputs.append(logits.unsqueeze(0))
                cls_loss += self.criterion(logits, train_y)
            # 所有组数据的平均分类损失函数为训练数据的分类损失函数
            cls_loss /= len(Xs)

            # 计算一致性损失
            consist_loss = self.GRAND.consistency_loss(outputs)

            # 合并分类损失与一致性损失
            loss = cls_loss + consist_loss

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
                # 获得最佳验证集准确率
                best_model = copy.deepcopy(self.model)
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
            dataset: Data, Data, 包含X, y, adjacency, test_mask,
                     train_mask和valid_mask
            split: string, 待预测的节点

            Output:
            -------
            accuracy: float, 节点分类准确率

        """

        # 模型推断模式
        self.model.eval()

        # 节点mask
        if split == 'train':
            mask = dataset.train_index
        elif split == 'valid':
            mask = dataset.valid_index
        else:  # split == 'test'
            mask = dataset.test_index

        # 获得待预测节点的输出
        # 推断过程中不包含DropNode, 仅有多阶聚合过程
        X = self.GRAND.random_propagate(
            adjacency=dataset.adjacency,
            X=dataset.X, train=False
        )[0]
        logits = self.model(X)
        predict_y = logits[mask].max(1)[1]

        # 计算预测准确率
        y = dataset.y[mask]
        accuracy = torch.eq(predict_y, y).float().mean()

        return accuracy
