"""SAGPool模型结构

    包含SAGPoolG结构和SAGPoolH结构

"""


import torch
import torch.nn as nn

from .utils import *
from .layers import *


class SAGPoolG(nn.Module):
    """SAGPoolG结构
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 keep_ratio, dropout, use_bias):
        """SAGPoolG结构

            Inputs:
            -------
            input_dim: int, 节点特征数量
            hidden_dim: int, 图卷积层计算输出的特征数
            output_dim: int, 输出类别数量
            keep_ratio: float, 图池化过程中每张图保留的topk节点所占比例
            dropout: float, 输出层使用的dopout比例
            use_bias: boolean, 图卷积层是否使用偏置

        """

        super(SAGPoolG, self).__init__()

        self.readout = Readout()
        self.act = nn.ReLU(inplace=True)

        # 图卷积层
        self.gcn1 = GraphConvolution(input_dim, hidden_dim, use_bias)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim, use_bias)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim, use_bias)

        # 图池化层
        self.sagpool = SelfAttentionPooling(hidden_dim * 3, keep_ratio)

        # 输出层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        return

    def forward(self, data):
        """SAGPoolG结构前馈

            Input:
            ------
            data: TUDataset, 包含x, edge_index和batch

            Output:
            -------
            logits: tensor, 各样本的logits

        """

        # 获得输入数据
        X, graph, batch = data.x, data.edge_index, data.batch

        # 计算初始邻接矩阵
        adjacency = generate_adjacency(X, graph)
        adjacency = normalize_adjacency(adjacency)
        adjacency = adjacency.to(X.device)

        # 拼接三层图卷积层的输出
        gcn1 = self.act(self.gcn1(adjacency, X))
        gcn2 = self.act(self.gcn2(adjacency, gcn1))
        gcn3 = self.act(self.gcn3(adjacency, gcn2))
        features = torch.cat([gcn1, gcn2, gcn3], dim=1)

        # 图池化
        mask_X, mask_adjacency, mask_batch = \
            self.sagpool(features, adjacency, batch)

        # 读出操作
        readout = self.readout(mask_X, mask_batch)
        # 计算输出
        logits = self.mlp(readout)
        return logits


class SAGPoolH(nn.Module):
    """SAGPoolH结构
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 keep_ratio, dropout, use_bias):
        """SAGPoolH结构

            Inputs:
            -------
            input_dim: int, 节点特征数量
            hidden_dim: int, 图卷积层计算输出的特征数
            output_dim: int, 输出类别数量
            keep_ratio: float, 图池化过程中每张图保留的topk节点所占比例
            dropout: float, 输出层使用的dopout比例
            use_bias: boolean, 图卷积层是否使用偏置

        """

        super(SAGPoolH, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.readout = Readout()

        # 图图卷积层和图池化层
        self.gcn1 = GraphConvolution(input_dim, hidden_dim, use_bias)
        self.sagpool1 = SelfAttentionPooling(hidden_dim, keep_ratio)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim, use_bias)
        self.sagpool2 = SelfAttentionPooling(hidden_dim, keep_ratio)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim, use_bias)
        self.sagpool3 = SelfAttentionPooling(hidden_dim, keep_ratio)

        # 输出层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        return

    def forward(self, data):
        """SAGPoolH结构前馈

            Input:
            ------
            data: TUDataset, 包含x, edge_index和batch

            Output:
            -------
            logits: tensor, 各样本的logits

        """

        # 获得输入数据
        X, graph, batch = data.x, data.edge_index, data.batch

        # 计算初始邻接矩阵
        adjacency = generate_adjacency(X, graph)
        adjacency = normalize_adjacency(adjacency)
        adjacency = adjacency.to(X.device)

        # 第1层: 图卷积 + 图池化 + 读出
        X = self.act(self.gcn1(adjacency, X))
        X, adjacency, batch = self.sagpool1(X, adjacency, batch)
        readout1 = self.readout(X, batch)

        # 第2层: 图卷积 + 图池化 + 读出
        X = self.act(self.gcn2(adjacency, X))
        X, adjacency, batch = self.sagpool2(X, adjacency, batch)
        readout2 = self.readout(X, batch)

        # 第3层: 图卷积 + 图池化 + 读出
        X = self.act(self.gcn3(adjacency, X))
        X, adjacency, batch = self.sagpool3(X, adjacency, batch)
        readout3 = self.readout(X, batch)

        # 融合三层读出结果计算输出
        readout = readout1 + readout2 + readout3
        logits = self.mlp(readout)
        return logits
