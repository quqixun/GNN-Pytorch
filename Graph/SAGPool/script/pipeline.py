"""SAGPool训练与预测
"""


import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score
from .model import SAGPoolG, SAGPoolH
from torch_geometric.data import DataLoader


class Pipeline(object):
    """SAGPool训练与预测
    """

    def __init__(self, model_name, **params):
        """FastGCN训练与预测

            加载GCN模型, 生成训练必要组件实例

            Input:
            ------
            params: dict, 模型参数和超参数, 格式为:
                    params = {
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
                            'weight_decay': 5e-4
                        }
                    }

        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__init_environment(params['random_state'])
        self.__build_model(model_name, **params['model'])
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

    def __build_model(self, model_name, **model_params):
        """加载模型

            Input:
            ------
            model_params: dict, 模型相关参数

        """

        assert model_name in ['SAGPoolG', 'SAGPoolH']

        model_class = SAGPoolG if model_name == 'SAGPoolG' else SAGPoolH
        self.model = model_class(**model_params)
        self.model.to(self.device)

        return

    def __build_components(self, **hyper_params):
        """加载组件

            Input:
            ------
            hyper_params: dict, 超参数

        """

        self.epochs = hyper_params['epochs']
        self.patience = hyper_params['patience']
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

    def train(self, dataset):
        """训练模型

            Input:
            ------
            dataset: Dataset

        """

        # 训练集batch加载器
        train_loader = DataLoader(
            dataset=dataset.train,
            batch_size=self.batch_size,
            shuffle=True
        )

        # 记录验证集效果最佳模型
        best_model = None

        # 记录验证集最佳准确率
        best_valid_f1 = 0

        # 获得最佳的验证集后计数轮次
        epochs_after_best = 0

        for epoch in range(self.epochs):
            # 模型训练模式
            self.model.train()

            # 用于记录每个epoch中所有batch的loss
            epoch_losses = []

            for i, data in enumerate(train_loader):
                # 模型输出
                data = data.to(self.device)
                logits = self.model(data)

                # 计算损失函数
                loss = self.criterion(logits, data.y)
                epoch_losses.append(loss.item())

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 计算epoch中所有batch的loss的均值
            epoch_loss = np.mean(epoch_losses)

            # 计算验证集loss和F1-Score
            valid_loss, valid_f1 = self.predict(dataset, 'valid')

            print('[Epoch:{:03d}]-[TrainLoss:{:.4f}]-[ValidLoss:{:.4f}]-[ValidF1:{:.4f}]'.format(
                epoch, epoch_loss, valid_loss, valid_f1))

            if valid_f1 >= best_valid_f1:
                best_model = copy.deepcopy(self.model)
                # 获得最佳验证集准确率
                best_valid_f1 = valid_f1
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
            eval_dataset = dataset.train
        elif split == 'valid':
            eval_dataset = dataset.valid
        else:  # split == 'test'
            eval_dataset = dataset.test

        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=1, shuffle=False
        )

        losses, y_true, y_pred = [], [], []
        for i, data in enumerate(eval_loader):
            # 模型输出
            data = data.to(self.device)
            logits = self.model(data)
            predict_y = logits.max(1)[1]

            y_true.append(predict_y.cpu().numpy()[0])
            y_pred.append(data.y.cpu().numpy()[0])
            losses.append(self.criterion(logits, data.y).item())

        loss = np.mean(losses)
        f1 = f1_score(y_true, y_pred, average='macro')
        return loss, f1
