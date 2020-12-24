"""定义GAT网络
"""


import torch
import torch.nn as nn

from .layers import GraphAttentionLayer


class GAT(nn.Module):
    """定义GAT网络 (dense input)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout, alpha):
        """定义GAT网络 (dense input)

            Inputs:
            -------
            input_dim: int, 输入维度
            hidden_dim: int, 隐层维度
            outut_dim: int, 输出维度
            num_heads: int, 多头注意力个数
            dropout: float, dropout比例
            alpha: float, LeakyReLU负数部分斜率

        """

        super(GAT, self).__init__()

        self.elu = nn.ELU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.attentions = [GraphAttentionLayer(input_dim, hidden_dim, dropout, alpha, True)
                           for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.output = GraphAttentionLayer(num_heads * hidden_dim, output_dim, dropout, alpha, False)

        return

    def forward(self, adjacency, X):
        """GAT网络 (dense input) 前馈

            Inputs:
            -------
            adjacency: tensor, 邻接矩阵
            X: tensor, 节点特征

            Output:
            -------
            output: tensor, 输出


        """

        out = self.dropout1(X)
        out = torch.cat([attention(adjacency, X) for attention in self.attentions], dim=1)
        out = self.dropout2(out)
        out = self.output(adjacency, out)
        out = self.log_softmax(self.elu(out))

        return out
