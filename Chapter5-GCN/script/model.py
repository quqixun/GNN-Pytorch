"""定义GCN模型
"""


import torch
import torch.nn as nn


class GraphConvolution(nn.Module):

    def __init__(self, adjacency, input_dim, output_dim, use_bias=True):
        """
        """

        super(GraphConvolution, self).__init__()

        self.use_bias = use_bias
        self.adjacency = adjacency
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.__init_parameters()

        return

    def __init_parameters(self, use_bias):
        """
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, X):
        """
        """

        support = torch.mm(X, self.weight)
        output = torch.sparse.mm(self.adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


class GCNet(nn.Module):
    """
    """

    def __init__(self, adjacency, input_dim, output_dim, hidden_dims, use_bias=True):
        """
        """

        super(GCNet, self).__init__()

        assert len(hidden_dims) >= 1

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                GraphConvolution(adjacency, dims[i], dims[i + 1], use_bias),
                nn.ReLU(inplace=True)
            ])

        layers.append(GraphConvolution(adjacency, dims[-1], output_dim, use_bias))
        self.models = nn.Sequential(layers)

        return

    def forward(self, X):
        """
        """

        logits = self.models(X)
        return logits
