"""定义Graph Attention层
"""


import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Dense GAT


class GraphAttentionLayer(nn.Module):
    """Graph Attention层 (dense input)
    """

    def __init__(self, input_dim, output_dim, dropout, alpha, bias=True):
        """Graph Attention层 (dense input)

            Inputs:
            -------
            input_dim: int, 输入维度
            outut_dim: int, 输出维度
            dropout: float, dropout比例
            alpha: float, LeakyReLU负数部分斜率
            bias: boolean, 是否使用偏置

        """

        super(GraphAttentionLayer, self).__init__()

        self.output_dim = output_dim

        # 节点特征线性变换
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # 邻接矩阵线性变换，计算attention
        self.a = nn.Parameter(torch.Tensor(2 * output_dim, 1))

        # 是否使用偏置
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        # 其他组件
        self.softmax = nn.Softmax(dim=1)
        self.dropout_X = nn.Dropout(p=dropout)
        self.dropout_Att = nn.Dropout(p=dropout)
        self.lrelu = nn.LeakyReLU(negative_slope=alpha)

        # 初始化参数
        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重
        """

        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        return

    def forward(self, X, edges):
        """Graph Attention层 (dense input) 前馈

            Inputs:
            -------
            X: tensor, 节点特征
            edges: tensor, 边的源节点与目标节点索引

            Output:
            -------
            h_prime: tensor, 输出

        """

        # 节点个数
        N = X.size(0)

        # 输入数据计算设备
        device = 'cuda' if X.is_cuda else 'cpu'

        # 节点特征线性变换
        X = self.dropout_X(X)
        Wh = torch.matmul(X, self.W)

        # 按邻接矩阵顺序, 组合一条边的两个节点特征
        Whij = torch.cat([Wh[edges[0]], Wh[edges[1]]], dim=1)

        # 计算attention
        e = self.lrelu(torch.matmul(Whij, self.a))
        attention = -9e15 * torch.ones([N, N], device=device, requires_grad=True)
        attention[edges[0], edges[1]] = e[:, 0]
        attention = self.dropout_Att(self.softmax(attention))

        # 更新节点特征
        h_prime = torch.matmul(attention, Wh)
        return h_prime


# ----------------------------------------------------------------------------
# Sparse GAT


class SparseGraphAttentionLayer(nn.Module):
    """Graph Attention层 (sparse input)
    """

    def __init__(self, input_dim, output_dim, dropout, alpha, bias=True):
        """Graph Attention层 (sparse input)

            Inputs:
            -------
            input_dim: int, 输入维度
            outut_dim: int, 输出维度
            dropout: float, dropout比例
            alpha: float, LeakyReLU负数部分斜率
            bias: boolean, 是否使用偏置

        """

        super(SparseGraphAttentionLayer, self).__init__()

        self.output_dim = output_dim

        # 节点特征线性变换
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # 邻接矩阵线性变换，计算attention
        self.a = nn.Parameter(torch.Tensor(2 * output_dim, 1))

        # 是否使用偏置
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        # 其他组件
        self.softmax = nn.Softmax(dim=1)
        self.dropout_X = nn.Dropout(p=dropout)
        self.dropout_Att = nn.Dropout(p=dropout)
        self.lrelu = nn.LeakyReLU(negative_slope=alpha)

        # 初始化参数
        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重
        """

        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        return

    def forward(self, X, edges):
        """Graph Attention层 (sparse input) 前馈

            Inputs:
            -------
            X: tensor, 节点特征
            edges: tensor, 边的源节点与目标节点索引

            Output:
            -------
            h_prime: tensor, 输出

        """

        # 节点个数
        N = X.size(0)

        # 输入数据计算设备
        device = 'cuda' if X.is_cuda else 'cpu'

        # 节点特征线性变换
        X = self.dropout_X(X)
        Wh = torch.matmul(X, self.W)

        # 按邻接矩阵顺序, 组合一条边的两个节点特征
        Whij = torch.cat([Wh[edges[0]], Wh[edges[1]]], dim=1)

        # 计算attention
        e = self.lrelu(torch.matmul(Whij, self.a))
        attention = self.__sparse_softmax(edges, e, N, device)
        attention = self.dropout_Att(self.softmax(attention))

        # 更新节点特征
        h_prime = self.__sparse_matmul(edges, attention, Wh)
        return h_prime

    @staticmethod
    def __sparse_softmax(edges, e, N, device):
        """稀疏数据的Softmax
        """

        source = edges[0]
        e_max = e.max()
        e_exp = torch.exp(e - e_max)
        e_exp_sum = torch.zeros(N, 1, device=device)
        e_exp_sum.scatter_add_(
            dim=0,
            index=source.unsqueeze(1),
            src=e_exp
        )
        e_exp_sum += 1e-10
        e_softmax = e_exp / e_exp_sum[source]

        return e_softmax

    @staticmethod
    def __sparse_matmul(edges, attention, Wh):
        """稀疏数据Matmul
        """

        source, target = edges
        h_prime = torch.zeros_like(Wh)
        h_prime.scatter_add_(
            dim=0,
            index=source.expand(Wh.size(1), -1).t(),
            src=attention * Wh[target]
        )

        return h_prime
