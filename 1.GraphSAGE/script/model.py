"""定义GraphSAGE模型
"""


import torch
import torch.nn as nn

from .aggregate import NeighborAggregator


class SAGEGCN(nn.Module):
    """聚合邻居特征用于更新节点特征
    """

    def __init__(self, input_dim, hidden_dim, activation='relu',
                 aggr_neighbor_method='mean', aggr_hidden_method='sum'):
        """
        """

        super(SAGEGCN, self).__init__()
        assert aggr_neighbor_method in ['mean', 'sum', 'max']
        assert aggr_hidden_method in ['sum', 'concat']
        assert activation in ['relu', None]

        self.aggr_hidden_method = aggr_hidden_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.activation = nn.ReLU(inplace=True) if activation is not None else None
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)

        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重和偏置
        """

        nn.init.kaiming_normal_(self.weight)

        return

    def forward(self, src_node_features, neighbor_node_features):
        """聚合邻居特征用于更新节点特征前馈
        """

        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden_method == 'sum':
            src_hidden = self_hidden + neighbor_hidden
        else:  # self.aggr_hidden_method == 'concat'
            src_hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)

        if self.activation is not None:
            src_hidden = self.activation(src_hidden)

        return src_hidden


class GraphSAGE(nn.Module):
    """
    """

    def __init__(self, input_dim, hidden_dims=[64, 64],
                 num_neighbors_list=[10, 10]):
        """
        """

        super(GraphSAGE, self).__init__()

        self.num_layers = len(hidden_dims)
        self.num_neighbors_list = num_neighbors_list
        self.gcn = [
            SAGEGCN(input_dim, hidden_dims[0], 'relu'),
            SAGEGCN(hidden_dims[0], hidden_dims[1], None)
        ]

        return

    def forward(self, node_feature_list):
        """
        """

        hidden = node_feature_list
        for i in range(self.num_layers):
            next_hidden = []

            for hop in range(self.num_layers - 1):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view(
                    src_node_num, self.num_neighbors_list[hop], -1
                )
                new_hidden = self.gcn[i](src_node_features, neighbor_node_features)
                next_hidden.append(new_hidden)

            hidden = next_hidden

        return hidden[0]
