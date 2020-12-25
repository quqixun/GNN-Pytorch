"""定义Attention层
"""


import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """Attention层 (dense input)
    """

    def __init__(self, input_dim, output_dim, dropout, alpha, activate=True):
        """Attention层 (dense input)

            Inputs:
            -------
            input_dim: int, 输入维度
            outut_dim: int, 输出维度
            dropout: float, dropout比例
            alpha: float, LeakyReLU负数部分斜率
            activate: boolean, 是否激活输出

        """

        super(GraphAttentionLayer, self).__init__()

        self.activate = activate
        self.output_dim = output_dim

        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.Tensor(2 * output_dim, 1))

        self.elu = nn.ELU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)
        self.lrelu = nn.LeakyReLU(negative_slope=alpha)

        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重
        """

        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.a)

        return

    def forward(self, adjacency, X):
        """Attention层 (dense input) 前馈

            Inputs:
            -------
            adjacency: tensor, 邻接矩阵
            X: tensor, 节点特征

            Output:
            -------
            h_prime: tensor, 输出

        """

        num_nodes = X.size(0)

        Wh = torch.mm(X, self.W)
        Whi = Wh.repeat_interleave(num_nodes, dim=0)
        Whj = Wh.repeat(num_nodes, 1)
        Whij = torch.cat([Whi, Whj], dim=1)
        Whij = Whij.view(num_nodes, num_nodes, 2 * self.output_dim)

        e = self.lrelu(torch.matmul(Whij, self.a).squeeze(2))
        attention = torch.where(adjacency > 0, e, -9e15 * torch.ones_like(e))
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)
        if self.activate:
            h_prime = self.elu(h_prime)

        return h_prime
