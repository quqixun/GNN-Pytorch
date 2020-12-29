"""定义邻居特征聚合方式
"""


import torch
import torch.nn as nn


class NeighborAggregator(nn.Module):
    """定义邻居特征聚合方式
    """

    def __init__(self, input_dim, output_dim, use_bias=True, aggr_method='mean'):
        """定义邻居特征聚合方式

            Inputs:
            -------
            input_dim: int, 输入特征维度
            output_dim: int, 输出特征维度
            use_bias: boolean, 是否使用偏置
            aggr_method: string, 聚合方式, 可选'mean', 'sum', 'max'

        """

        super(NeighborAggregator, self).__init__()
        assert aggr_method in ['mean', 'sum', 'max']

        self.use_bias = use_bias
        self.aggr_method = aggr_method

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重和偏置
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, neighbor_features):
        """节点聚合前馈

            Input:
            ------
            neighbor_features: tensor in shape [num_nodes, input_dim],
                              节点的邻居特征

            Output:
            -------
            neighbor_hidden: tensor in shape [num_nodes, output_dim],
                             聚合邻居特征后的节点特征

        """

        # 聚合邻居特征
        if self.aggr_method == 'mean':
            aggr_neighbor = neighbor_features.mean(dim=1)
        elif self.aggr_method == 'sum':
            aggr_neighbor = neighbor_features.sum(dim=1)
        else:  # self.aggr_method == 'max'
            aggr_neighbor = neighbor_features.max(dim=1)

        # 线性变换获得隐层特征
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden
