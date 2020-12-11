"""定义GCN模型
"""


import torch
import torch.nn as nn


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, use_bias=True):
        """
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
        """
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, adjacency, X):
        """
        """

        support = torch.mm(X, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


class GCNet(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, hidden_dim, use_bias=True):
        """
        """

        super(GCNet, self).__init__()

        self.gcn1 = GraphConvolution(input_dim, hidden_dim, use_bias)
        self.act1 = nn.ReLU(inplace=True)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim, use_bias)

        return

    def forward(self, adjacency, X):
        """
        """

        out = self.act1(self.gcn1(adjacency, X))
        logits = self.gcn2(adjacency, out)
        return logits
