"""定义Attention层
"""


import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Dense Input


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

        # 节点特征线性变换
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # 邻接矩阵线性变换，计算attention
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

        # 节点个数
        num_nodes = X.size(0)

        # 节点特征线性变换
        Wh = torch.mm(X, self.W)

        # 按邻接矩阵顺序, 组合一条边的两个节点特征
        # Whi: n1, n1, ..., n1, n2, n2, ..., n2, nN, nN, ..., nN
        #      |<--   N   -->|  |<--   N   -->|  |<--   N   -->|
        # Whj: n1, n2, ..., nN, n1, n2, ..., nN, ..., n1, n2, ..., nN,
        #      |<--   N   -->|  |<--   N   -->|       |<--   N   -->|
        Whi = Wh.repeat_interleave(num_nodes, dim=0)
        Whj = Wh.repeat(num_nodes, 1)
        Whij = torch.cat([Whi, Whj], dim=1)
        Whij = Whij.view(num_nodes, num_nodes, 2 * self.output_dim)

        # 计算attention
        e = self.lrelu(torch.matmul(Whij, self.a).squeeze(2))
        attention = torch.where(adjacency > 0, e, -9e15 * torch.ones_like(e))
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # 更新节点特征
        h_prime = torch.matmul(attention, Wh)
        if self.activate:
            h_prime = self.elu(h_prime)

        return h_prime


# ----------------------------------------------------------------------------
# Sparse Input


class SparseMMFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, edges, attentions, N, X):
        """
        """

        assert not edges.requires_grad

        ctx.N = N
        shape = torch.Size([N, N])

        adjacency = torch.sparse_coo_tensor(edges, attentions, shape)
        ctx.save_for_backward(adjacency, X)

        return torch.matmul(adjacency, X)

    @staticmethod
    def backward(ctx, grad_output):
        """
        """

        adjacency, X = ctx.saved_tensors
        grad_adj = grad_X = None

        if ctx.needs_input_grad[1]:
            grad_adj = grad_output.matmul(X.t())
            indices = adjacency._indices()
            edges = indices[0, :] * ctx.N + indices[1, :]
            grad_adj = grad_adj.view(-1)[edges]

        if ctx.needs_input_grad[3]:
            grad_X = adjacency.t().matmul(grad_output)

        return None, grad_adj, None, grad_X


class SparseMM(nn.Module):

    def forward(self, edges, attentions, N, X):
        return SparseMMFunc(edges, attentions, N, X)


class SparseGraphAttentionLayer(nn.Module):
    """Attention层 (sparse input)
    """

    def __init__(self, input_dim, output_dim, dropout, alpha, activate=True):
        """Attention层 (sparse input)

            Inputs:
            -------
            input_dim: int, 输入维度
            outut_dim: int, 输出维度
            dropout: float, dropout比例
            alpha: float, LeakyReLU负数部分斜率
            activate: boolean, 是否激活输出

        """

        super(SparseGraphAttentionLayer, self).__init__()

        self.activate = activate
        self.output_dim = output_dim

        # 节点特征线性变换
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # 邻接矩阵线性变换，计算attention
        self.a = nn.Parameter(torch.Tensor(1, 2 * output_dim))

        self.spmm = SparseMM()
        self.elu = nn.ELU(inplace=True)
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
        """Attention层 (sparse input) 前馈

            Inputs:
            -------
            adjacency: tensor, 邻接矩阵
            X: tensor, 节点特征

            Output:
            -------
            h_prime: tensor, 输出

        """

        N = X.size(0)
        edges = torch.nonzero(adjacency, as_tuple=False).t()
        device = 'cuda' if X.is_cuda else 'cpu'
        row = torch.ones(size=(N, 1), device=device)

        # 节点特征线性变换
        Wh = torch.mm(X, self.W)

        # 按邻接矩阵顺序, 组合一条边的两个节点特征
        Whij = torch.cat([Wh[edges[0, :]], Wh[edges[1, :]]], dim=1).t()

        print(self.a.size(), Whij.size())

        # 计算attention
        e = torch.exp(-self.lrelu(self.a.mm(Whij)).squeeze())
        e_rowsum = self.spmm(edges, e, N, row)
        e = self.dropout(e)

        # 更新节点特征
        h_prime = self.spmm(edges, e, N, Wh)
        h_prime = h_prime.div(e_rowsum)
        if self.activate:
            h_prime = self.elu(h_prime)

        return h_prime
