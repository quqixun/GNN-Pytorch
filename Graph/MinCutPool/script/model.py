"""MinCutPool模型结构
"""


import torch.nn as nn

from .utils import *
from .layers import *
from math import ceil
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch


class MinCutPoolModel(nn.Module):
    """MinCutPool模型结构
    """

    def __init__(self, input_dim, hidden_dim, output_dim, avg_nodes,
                 aggregate='sum', dropout=0.0, use_bias=True):
        """MinCutPool模型结构

            Inputs:
            -------
            input_dim: int, 节点特征数量
            hidden_dim: int, 各隐层计算输出的特征数
            output_dim: int, 输出类别数量
            avg_nodes: int, 所有图平均节点数, 用于确定各聚类层的聚类个数
            aggregate: string, DenseGraphConvolution层中节点聚合方式
            dropout: float, 输出层使用的dopout比例
            use_bias: boolean, 图卷积层是否使用偏置

        """

        super(MinCutPoolModel, self).__init__()

        self.act = nn.ReLU(inplace=True)

        # 图卷积层
        self.conv1 = GraphConvolution(input_dim, hidden_dim, use_bias)
        self.conv2 = DenseGraphConvolution(hidden_dim, hidden_dim, aggregate, use_bias)
        self.conv3 = DenseGraphConvolution(hidden_dim, hidden_dim, aggregate, use_bias)

        # 节点聚类层
        self.cluster1 = nn.Linear(hidden_dim, ceil(0.5 * avg_nodes))
        self.cluster2 = nn.Linear(hidden_dim, ceil(0.25 * avg_nodes))

        # 池化层
        self.mincutpool = MinCutPooling()

        # 输出层
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        return

    def forward(self, data):
        """MinCutPool模型前馈

            Input:
            ------
            data: TUDataset, 包含x, edge_index和batch

            Output:
            -------
            logits: tensor, 各样本的logits
            loss: float tenor scalar, mincut loss + orthogonality loss

        """

        # 获得输入数据
        # Na -> 所有图包含节点的总数
        # Ne -> 所有图包含边的总数
        # F  -> 节点特征数量
        # X:     [Na, F], 所有图包含的节点的特征
        # batch: [Na],    指明每个节点所属的图
        # graph: [2, Ne], 所有图包含的边
        X, graph, batch = data.x, data.edge_index, data.batch

        # 将输入数据处理成batch格式
        # b    -> batch size
        # Nmax -> 包含最多节点的图的节点数量
        # F    -> 节点特征数量
        # X:         [b, Nmax, F],    每张图包含的节点的特征
        # batch:     [b, Nmax],       每张图中的有效节点索引
        # adjacency: [b, Nmax, Nmax], 每张图的邻接矩阵
        adjacency = to_dense_adj(graph, batch)
        x0, batch = to_dense_batch(X, batch)
        a0 = batch_normalize_adjacency(adjacency, batch)

        # 第1层聚类结果
        x1 = self.act(self.conv1(a0, x0))
        s1 = self.cluster1(x1)
        x1, a1, loss1 = self.mincutpool(x1, a0, s1, batch)

        # 第2层聚类结果
        x2 = self.act(self.conv2(a1, x1))
        s2 = self.cluster2(x2)
        x2, a2, loss2 = self.mincutpool(x2, a1, s2)

        # 输出层
        x3 = self.act(self.conv3(a2, x2).mean(dim=1))
        logits = self.mlp(x3)

        # 合并各层loss
        loss = loss1 + loss2
        return logits, loss
