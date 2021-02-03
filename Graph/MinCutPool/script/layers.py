"""MinCutPool网络相关层函数
"""


import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 图卷积层


class GraphConvolution(nn.Module):
    """图卷积层
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积层

            Inputs:
            -------
            input_dim: int, 输入特征维度
            output_dim: int, 输出特征维度
            use_bias: boolean, 是否使用偏置

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
        """初始化权重和偏置
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, adjacency, X):
        """图卷积层前馈

            Inputs:
            -------
            adjacency: tensor in shape [batch_size, num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [batch_size, num_nodes, input_dim], 节点特征

            Output:
            -------
            output: tensor in shape [batch_size, num_nodes, output_dim], 输出

        """

        weight = torch.cat([self.weight.unsqueeze(0)] * X.size(0))
        support = torch.bmm(X, weight)
        output = torch.bmm(adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


# ----------------------------------------------------------------------------
# DenseGraphConv层


class DenseGraphConvolution(nn.Module):
    """DenseGraphConv层

        定义节点特征聚合方式

    """

    def __init__(self, input_dim, output_dim, aggregate='sum', use_bias=True):
        """DenseGraphConv层
        """

        super(DenseGraphConvolution, self).__init__()

        assert aggregate in ['sum', 'max', 'mean'], 'unknown aggregate'
        self.aggregate = aggregate

        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.linear2 = nn.Linear(input_dim, output_dim, bias=use_bias)

        return

    def forward(self, adjacency, X):
        """DenseGraphConv层前馈
        """

        # self.aggregate == 'sum'
        output = self.linear1(torch.matmul(adjacency, X))

        if self.aggregate == 'max':
            output = output.max(dim=-1, keepdim=True)[0]
        elif self.aggregate == 'mean':
            output = output / output.sum(dim=-1, keepdim=True)
        else:
            pass

        output = output + self.linear2(X)

        return output


# ----------------------------------------------------------------------------
# MinCutPooling层


class MinCutPooling(nn.Module):
    """
    """

    def forward(self, X, A, S, batch=None):
        """
        """

        device = X.device

        num_clusters = S.size(-1)
        S = torch.softmax(S, dim=-1)

        if batch is not None:
            batch = batch.unsqueeze(-1) * 1.0
            X, S = X * batch, S * batch

        # mincut loss
        X_pool = torch.matmul(S.transpose(1, 2), X)
        A_pool = torch.matmul(torch.matmul(S.transpose(1, 2), A), S)
        # mincut_num = [torch.trace(n).item() for n in A_pool]
        # mincut_num = torch.FloatTensor(mincut_num).to(device)
        mincut_num = torch.einsum('ijj->i', A_pool)

        D_flat = torch.sum(A, dim=-1)
        # D = torch.cat([torch.diag(d).unsqueeze(0) for d in D_flat]).to(device)
        D_eye = torch.eye(D_flat.size(1)).type_as(A).to(device)
        D_flat = D_flat.unsqueeze(2).expand(*D_flat.size(), D_flat.size(1))
        D = D_eye * D_flat

        D_pool = torch.matmul(torch.matmul(S.transpose(1, 2), D), S)
        # mincut_den = [torch.trace(d).item() for d in D_pool]
        # mincut_den = torch.FloatTensor(mincut_den).to(device)
        mincut_den = torch.einsum('ijj->i', D_pool)
        mincut_loss = -1 * torch.mean(mincut_num / mincut_den)

        # orthogonality loss
        St_S = torch.matmul(S.transpose(1, 2), S)
        I_S = torch.eye(num_clusters).to(device)

        St_S_norm = St_S / torch.norm(St_S, dim=(-1, -2), keepdim=True)
        I_S_norm = I_S / torch.norm(I_S)
        ortho_loss = torch.norm(St_S_norm - I_S_norm, dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        # 正则化A_pool
        # 将A_pool中每个邻接矩阵的对角线元素置0
        index = torch.arange(num_clusters, device=device)
        A_pool[:, index, index] = 0
        D_pool = torch.sum(A_pool, dim=-1)
        D_pool = torch.sqrt(D_pool)[:, None] + 1e-15
        A_pool = (A_pool / D_pool) / D_pool.transpose(1, 2)

        return X_pool, A_pool, mincut_loss + ortho_loss
