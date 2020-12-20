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
        """聚合邻居特征用于更新节点特征

            Inputs:
            -------
            input_dim: int, 输入特征的维度
            hidden_dim: int, 隐层特征的维度
            activation: string or None, 激活函数, ['relu', None]
            aggr_neighbor_method: 邻居特征聚合方法，['mean', 'sum', 'max']
            aggr_hidden_method: 节点特征的更新方法，['sum', 'concat']

        """

        super(SAGEGCN, self).__init__()

        # 检查输入
        assert aggr_neighbor_method in ['mean', 'sum', 'max']
        assert aggr_hidden_method in ['sum', 'concat']
        assert activation in ['relu', None]

        self.aggr_hidden_method = aggr_hidden_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.activation = nn.ReLU(inplace=True) if activation is not None else None

        # 邻居节点特征聚合器
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

            Inputs:
            -------
            src_node_features: tensor, 源节点特征
            neighbor_node_features: tensor, 邻居节点特征

            Output:
            -------
            src_hidden: tensor, 更新后的源节点隐层特征

        """

        # 聚合邻居节点特征
        neighbor_hidden = self.aggregator(neighbor_node_features)

        # 源节点本身特征的线性变换
        self_hidden = torch.matmul(src_node_features, self.weight)

        # 合并源节点特征与邻居节点特征
        if self.aggr_hidden_method == 'sum':
            src_hidden = self_hidden + neighbor_hidden
        else:  # self.aggr_hidden_method == 'concat'
            src_hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)

        if self.activation is not None:
            # 最后输出层前各隐层的激活函数
            src_hidden = self.activation(src_hidden)

        return src_hidden


class GraphSAGE(nn.Module):
    """GraphSAGE模型
    """

    def __init__(self, input_dim, hidden_dims=[64, 64],
                 num_neighbors_list=[10, 10]):
        """GraphSAGE模型

            Inputs:
            -------
            input_dim: int, 输入特征的维度
            hidden_dims: list of ints, 每一隐层输出特征的维度
            num_neighbors_list: list of ints, 每一阶邻居采样个数

        """

        super(GraphSAGE, self).__init__()

        self.num_layers = len(num_neighbors_list)
        self.num_neighbors_list = num_neighbors_list

        # 定义每一阶的邻居节点特征聚合层
        self.gcn = nn.ModuleList()
        self.gcn.append(SAGEGCN(input_dim, hidden_dims[0], 'relu'))
        for index in range(0, len(hidden_dims) - 2):
            self.gcn.append(SAGEGCN(hidden_dims[index], hidden_dims[index + 1], 'relu'))
        self.gcn.append(SAGEGCN(hidden_dims[-2], hidden_dims[-1], None))

        return

    def forward(self, node_feature_list):
        """GraphSAGE模型前馈

            Input:
            ------
            node_feature_list: list of tensor, 每一阶邻居节点特征列表

            Output:
            -------
            logits: tensor, 模型输出

        """

        # 每一层邻居节点特征列表, 第一个tensor为源节点特征
        hidden = node_feature_list

        for i in range(self.num_layers):
            # 记录每一隐层网络输出
            next_hidden = []

            # 每一隐层网络
            gcn = self.gcn[i]

            for hop in range(self.num_layers - i):
                # 每一阶源节点特征
                src_node_features = hidden[hop]

                # 每一阶邻居节点特征
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view((
                    src_node_num, self.num_neighbors_list[hop], -1
                ))

                # 聚合邻居节点特征并计算源节点隐层特征
                new_hidden = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(new_hidden)

            hidden = next_hidden

        # 源节点logits作为输出
        logits = hidden[0]
        return logits
