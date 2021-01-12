"""节点采样器

    对于每个batch产生的源节点，从所有目标节点中进行重要性采样，
    重要性因子正比于目标节点的degree，在计算时利用源节点和采样
    的目标节点之间形成的子图进行损失和梯度的计算

"""


import numpy as np
import scipy.sparse as sp

from .utils import sparse_matrix_to_tensor
from scipy.sparse.linalg import norm as sparse_norm


class Sampler(object):
    """节点采样器

        对于每个batch产生的源节点，从所有目标节点中进行重要性采样，
        重要性因子正比于目标节点的degree，在计算时利用源节点和采样
        的目标节点之间形成的子图进行损失和梯度的计算

    """

    def __init__(self, input_dim, sampler_dims, device):
        """节点采样器

            Inputs:
            -------
            input_dim: int, 节点特征维度
            sampler_dims: list of ints, 每一层网络采样的节点个数
            device: string, 使用的计算设备

        """

        self.device = device
        self.input_dim = input_dim
        self.sampler_dims = sampler_dims
        self.num_layers = len(sampler_dims)

        return

    def sampling(self, X, adjacency, batch_nodes):
        """基于重要性的节点采样

            Inputs:
            -------
            X: tensor, 节点特征
            adjacency: sparse numpy array, 初始邻接矩阵
            batch_nodes: numpy array, 每个batch产生的源节点索引列表

            Outputs:
            --------
            sampled_X: tensor, 第一层网络中采样的节点特征
            sampled_adjacency: list of sparse tensor, 用于每一层网络的采样的邻接矩阵

            Example:
            --------
            X in shape (num_nodes, num_feats)
            adjacency in shape (num_nodes, num_nodes)
            batch_nodes in shape (batch_size,)
            sampler_dims is [dim1, dim2)]

            sampled_X in shape (dim2, num_feats)
            sampled_adjacency is [
                adjacency1 in shape (dim1, dim2),
                adjacency2 in shape (batch_size, dim1)
            ]

        """

        self.X = X
        self.N = X.shape[0]
        self.adjacency = adjacency
        # 计算目标节点(每列)对于每个源节点(每行)的重要性(归一化)
        self.__init_probability(adjacency)

        # 用于记录每层网络的邻接矩阵
        sampled_adjacency = []

        # 第一层网络中采样的节点特征
        sampled_nodes = batch_nodes

        for layer in range(self.num_layers - 1, -1, -1):
            # 每层网络采样节点和邻接矩阵
            nodes, adjacency = self.__single_layer_sampling(
                sampled_nodes, self.sampler_dims[layer])

            # 将每层采样出的邻接矩阵转换为稀疏tensor
            adjacency = sparse_matrix_to_tensor(adjacency, self.device)
            sampled_adjacency.append(adjacency)
            # 下一层网络使用采样的源节点
            sampled_nodes = nodes

        # 第一层网络采样后的节点特征
        sampled_X = self.X[sampled_nodes]
        # 反向，目的是使最终网络输出为batch的logits
        sampled_adjacency.reverse()

        return sampled_X, sampled_adjacency

    def __init_probability(self, adjacency):
        """计算目标节点(每列)对于每个源节点(每行)的重要性(归一化)

            Input:
            ------
            adjacency: sparse numpy array, 初始邻接矩阵

        """

        norm_adjacency = sparse_norm(adjacency, axis=0)
        self.probability = norm_adjacency / np.sum(norm_adjacency)

        return

    def __single_layer_sampling(self, nodes, sampler_dim):
        """对每一层网络采样目标节点和邻接矩阵

            Input:
            ------
            nodes: numpy array， 采样出的源节点索引列表
            sampler_dim: int, 需采样的目标节点个数

            Outputs:
            --------
            sampled_nodes: numpy array, 采样的目标节点索引列表
            sampled_adjacency: sparse numpy array, 采样的邻接矩阵

        """

        # 采样的源节点的邻接矩阵, 使用初始邻接矩阵相应的行
        adjacency = self.adjacency[nodes, :]

        # 对源节点所有可用的目标节点计算重要性
        neighbors = np.nonzero(np.sum(adjacency, axis=0))[1]
        probability = self.probability[neighbors]
        probability = probability / np.sum(probability)

        # 对目标节点按重要性采样
        sampled = np.random.choice(
            np.arange(np.size(neighbors)),
            size=sampler_dim,
            replace=True,
            p=probability
        )

        # 获得采样后的目标节点索引列表
        sampled_nodes = neighbors[sampled]

        # 获得采样后的由源节点和目标节点组成的邻接矩阵
        sampled_adjacency = adjacency[:, sampled_nodes]
        # 邻接矩阵归一化
        sampled_probability = probability[sampled]
        sampled_adjacency = sampled_adjacency.dot(sp.diags(
            1.0 / (sampled_probability * sampler_dim)
        ))

        return sampled_nodes, sampled_adjacency
