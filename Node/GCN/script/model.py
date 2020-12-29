"""定义GCN模型

    定义图卷积层和简单图网络。

"""


import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """图卷积层
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积层

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


class GCNet(nn.Module):
    """简单图卷积网络

        定义包含两层图卷积的简单网络。

    """

    def __init__(self, input_dim, output_dim, hidden_dim, use_bias=True):
        """简单图卷积网络

            Inputs:
            -------
            input_dim: int, 节点特征维度
            output_dim: int, 节点类别数
            hidden_dim: int, 第一层图卷积输出维度
            use_bias: boolean, 是否使用偏置

        """

        super(GCNet, self).__init__()

        self.gcn1 = GraphConvolution(input_dim, hidden_dim, use_bias)
        self.act1 = nn.ReLU(inplace=True)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim, use_bias)

        return

    def forward(self, adjacency, X):
        """简单图卷积网络前馈

            Inputs:
            -------
            adjacency: tensor in shape [num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [num_nodes, input_dim], 节点特征

            Output:
            -------
            logits: tensor in shape [num_nodes, output_dim], 输出

        """

        out = self.act1(self.gcn1(adjacency, X))
        logits = self.gcn2(adjacency, out)
        return logits
