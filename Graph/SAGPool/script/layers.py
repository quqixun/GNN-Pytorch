"""SAGPool网络相关层函数
"""


import torch
import torch.nn as nn

from .utils import topk, filter_adjacency
from torch_scatter import scatter_max, scatter_mean


# ----------------------------------------------------------------------------
# 图卷积层


class GraphConvolution(nn.Module):
    """图卷积层
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积层

            SAGPool中使用图卷积层计算每个图中每个节点的score

            Inputs:
            -------
            input_dim: int, 输入特征维度
            output_dim: int, 输出特征维度
            use_bias: boolean, 是否使用偏置

        """

        super(GraphConvolution, self).__init__()

        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重和偏置
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, adjacency, X):
        """图卷积层前馈

            Inputs:
            -------
            adjacency: tensor in shape [num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [num_nodes, input_dim], 节点特征

            Output:
            -------
            output: tensor in shape [num_nodes, output_dim], 输出

        """

        support = torch.mm(X, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


# ----------------------------------------------------------------------------
# 读出操作


def global_max_pool(X, graph_indicator):
    """全局最大值池化

        计算图中所有节点特征的全局最大值作为图特征

        Inputs:
        -------
        X: tensor, 所有图所有节点(可能是池化后)的特征
        graph_indicator: tensor, 指明每个节点所属的图

        Output:
        -------
        max_pool: tensor, 全局最大值池化后的图特征

    """

    num_graphs = graph_indicator.max().item() + 1
    max_pool = scatter_max(X, graph_indicator, dim=0, dim_size=num_graphs)[0]

    return max_pool


def global_avg_pool(X, graph_indicator):
    """全局平均值池化

        计算图中所有节点特征的全局平均值作为图特征

        Inputs:
        -------
        X: tensor, 所有图所有节点(可能是池化后)的特征
        graph_indicator: tensor, 指明每个节点所属的图

        Output:
        -------
        avg_pool: tensor, 全局平均值池化后的图特征

    """

    num_graphs = graph_indicator.max().item() + 1
    avg_pool = scatter_mean(X, graph_indicator, dim=0, dim_size=num_graphs)

    return avg_pool


class Readout(nn.Module):
    """图读出操作
    """

    def forward(self, X, graph_indicator):
        """图读出操作前馈

            拼接每个图的全局最大值特征和全局平局值特征作为图特征

            Inputs:
            -------
            X: tensor, 所有图所有节点(可能是池化后)的特征
            graph_indicator: tensor, 指明每个节点所属的图

            Output:
            -------
            readout: tensor, 读出操作获得的图特征

            """

        readout = torch.cat([
            global_avg_pool(X, graph_indicator),
            global_max_pool(X, graph_indicator)
        ], dim=1)

        return readout


# ----------------------------------------------------------------------------
# 自注意力机制池化层


class SelfAttentionPooling(nn.Module):
    """自注意力机制池化层
    """

    def __init__(self, input_dim, keep_ratio):
        """自注意力机制池化层

            使用GCN计算每个图中的每个节点的score作为重要性,
            筛选每个图中topk个重要的节点, 获取重要节点的邻接矩阵,
            使用重要节点特征和邻接矩阵用于后续操作

            Inputs:
            -------
            input_dim: int, 输入的节点特征数量
            keep_ratio: float, 每个图中topk的节点占所有节点的比例

        """

        super(SelfAttentionPooling, self).__init__()

        self.keep_ratio = keep_ratio
        self.act = nn.Tanh()
        self.gcn = GraphConvolution(input_dim, 1)

        return

    def forward(self, X, adjacency, graph_batch):
        """自注意力机制池化层前馈

            Inputs:
            -------
            X: tensor, 节点特征
            adjacency: tensor, 输入节点构成的邻接矩阵
            graph_nbatch: tensor, 指明每个节点所属的图

        """

        # 计算每个图中每个节点的重要性
        node_score = self.gcn(adjacency, X)
        node_score = self.act(node_score)

        # 获得每个途中topk和重要节点
        mask = topk(node_score, graph_batch, self.keep_ratio)
        # 获得重要节点特征, 指明重要节点所属的图, 生成由重要节点构成的邻接矩阵
        mask_X = X[mask] * node_score.view(-1, 1)[mask]
        mask_graph_batch = graph_batch[mask]
        mask_adjacency = filter_adjacency(adjacency, mask)

        return mask_X, mask_adjacency, mask_graph_batch
