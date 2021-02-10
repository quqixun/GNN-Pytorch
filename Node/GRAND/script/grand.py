"""GRAND的相关操作

    1. Augmentation过程，DropNode + Mised-Order Propagation,
       从一组训练数据中生成多组数据用于预测
    2. 计算一致性损失, 一致性损失越小，多组数据的预测结果越接近

"""

import copy
import torch
import numpy as np


class GRAND(object):
    """GRAND的相关操作

        1. Augmentation过程，DropNode + Mised-Order Propagation,
           从一组训练数据中生成多组数据用于预测
        2. 计算一致性损失, 一致性损失越小，多组数据的预测结果越接近

    """

    def __init__(self, **grand_params):
        """GRAND的相关操作

            1. Augmentation过程，DropNode + Mised-Order Propagation,
               从一组训练数据中生成多组数据用于预测
            2. 计算一致性损失, 一致性损失越小，多组数据的预测结果越接近

        """

        self.S = grand_params['S'] # Augmentation次数
        self.K = grand_params['K'] # Order聚合次数
        self.D = grand_params['D'] # DropNode比例
        self.T = grand_params['T'] # Temperature控制类别分布
        self.L = grand_params['L'] # Consistensy Loss系数

        return

    def random_propagate(self, adjacency, X, train=False):
        """数据Augmentation

            使用DropNode + Mised-Order Propagation对训练数据做
            Augmentation, 生成多组数据用于后续模型训练

            Inputs:
            -------
            adjacency: tensor in shape [num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [num_nodes, num_feats], 节点特征
            train: boolean, True:训练过程, False:推断过程

        """

        device = X.device
        num_nodes = X.size(0)

        # 推断过程不做DropNode, 仅使用原数据心境聚合操作
        S = self.S if train else 1

        # 用于记录Augmentation后产生的多组数据
        Xs = []
        for _ in range(S):
            Xcp = copy.deepcopy(X)
            # DropNode, 随机讲某些节点特征置0, 仅对训练数据操作
            if train:
                drop_nodes = torch.FloatTensor(np.ones(num_nodes) * self.D)
                masks = torch.bernoulli(1 - drop_nodes).unsqueeze(1)
                Xcp = masks.to(device) * Xcp  # * (1 / (1 - self.D))
            else:  # scale推断数据
                Xcp *= (1 - self.D)

            # Mixed-order propagation, 聚合操作
            Xk = copy.deepcopy(Xcp)
            for _ in range(self.K):
                Xcp = torch.spmm(adjacency, Xcp).detach()
                Xk += Xcp
            Xk = (Xk / (self.K + 1)).detach()
            Xs.append(Xk)

        # 训练过程: Xs包含self.S个tensor, 为训练数据Augmentation后的结果
        # 推断过程: Xs仅包含1个tensor
        return Xs

    def consistency_loss(self, pred):
        """计算一致性损失

            计算不同Augmentation数据间预测的一致性损失,
            一致性损失越小, 说明不同Augmentation数据间预测结果越接近

            Input:
            ------
            pred: list of tensors, Augmentation后每组数据的预测logits

            Output:
            -------
            consist_loss: torch loss, 一致性损失

        """

        pred = torch.cat(pred, dim=0)
        pred = torch.softmax(pred, dim=-1)
        pred_avg = torch.mean(pred, dim=(0, 1), keepdim=True)

        pred_pow = torch.pow(pred_avg, 1.0 / self.T)
        pred_sharp = pred_pow / torch.sum(pred_pow)
        pred_sharp = pred_sharp.detach()
        consist_loss = self.L * (pred - pred_sharp).pow(2).mean()

        return consist_loss
