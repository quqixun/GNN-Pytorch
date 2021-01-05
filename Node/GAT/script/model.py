"""定义GAT网络
"""


import torch
import torch.nn as nn

from .layers import GraphAttentionLayer, SparseGraphAttentionLayer


class GAT(nn.Module):
    """定义GAT网络
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout, alpha, sparse=False):
        """定义GAT网络

            Inputs:
            -------
            input_dim: int, 输入维度
            hidden_dim: int, 隐层维度
            outut_dim: int, 输出维度
            num_heads: int, 多头注意力个数
            dropout: float, dropout比例
            alpha: float, LeakyReLU负数部分斜率

        """

        print('sparse', sparse)

        super(GAT, self).__init__()

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        if sparse:
            # 使用稀疏数据的attention层
            attention_layer = SparseGraphAttentionLayer
        else:
            # 使用稠密数据的attention层
            attention_layer = GraphAttentionLayer

        # 多头注意力层
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(attention_layer(input_dim, hidden_dim, dropout, alpha, True))

        self.output = attention_layer(num_heads * hidden_dim, output_dim, dropout, alpha, False)

        return

    def forward(self, adjacency, X):
        """GAT网络前馈

            Inputs:
            -------
            adjacency: tensor, 邻接矩阵
            X: tensor, 节点特征

            Output:
            -------
            output: tensor, 输出

        """

        out = self.dropout1(X)
        # 拼接多头注意力层输出
        out = torch.cat([attention(adjacency, out) for attention in self.attentions], dim=1)
        out = self.dropout2(out)
        output = self.output(adjacency, out)

        return output
