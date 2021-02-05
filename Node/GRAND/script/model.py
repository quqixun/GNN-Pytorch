"""Simple MLP模型
"""


import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP模型
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, use_bn=False):
        """Simple MLP模型

            Inputs:
            -------
            input_dim: int, 输入的节点特征维度
            hidden_dim: int, 隐层特征维度
            output_dim: int, 输出的类别数
            dropout: float, dropout比率
            use_bn: boolean, 是否使用batch normalization

        """

        super(MLP, self).__init__()

        self.use_bn = use_bn
        self.act = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        return

    def forward(self, X):
        """Simple MLP模型前馈

            Input:
            ------
            X: tensor in shape [num_nodes, inpt_dims], 输入节点特征

            Output:
            -------
            logits: tensor in shape [num_nodes, output_dim], 输出

        """

        if self.use_bn:
            X = self.bn1(X)
        out = self.dropout1(X)
        out = self.act(self.linear1(out))

        if self.use_bn:
            out = self.bn2(out)
        out = self.dropout2(out)
        logits = self.linear2(out)

        return logits
