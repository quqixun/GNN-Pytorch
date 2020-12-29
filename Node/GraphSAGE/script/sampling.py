"""根据节点进行邻居采样

    根据源节点采样指定数量的邻居节点, 使用的是有放回的采样,
    某个节点的邻居节点数量少于采样数量时, 采样结果出现重复的节点

"""


import numpy as np


def sampling(src_nodes, sample_num, neighbor_dict):
    """根据源节点进行一阶采样

        Inputs:
        -------
        src_nodes: list or numpy array, 源节点列表
        sample_num: int, 需采样的邻居节点数量
        neighbor_dict: dict, 节点到其邻居节点的映射表

        Output:
        -------
        sampling_results: numpy array, 采样后的节点列表

    """

    sampling_results = []
    for node in src_nodes:
        # 从节点的邻居中进行有放回采样
        sample = np.random.choice(neighbor_dict[node], size=(sample_num,))
        sampling_results.append(sample)

    return np.asarray(sampling_results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_dict):
    """根据源节点进行多阶采样

        Inputs:
        -------
        src_nodes: list or numpy array, 源节点列表
        sample_nums: list of ints, 每一阶需采样的邻居节点数量
        neighbor_dict: dict, 节点到其邻居节点的映射表

        Output:
        -------
        sampling_results: list of numpy array, 每一阶采样后的节点列表

    """

    sampling_results = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        # 对每一阶进行邻居采样
        hopk_sampling = sampling(sampling_results[k], hopk_num, neighbor_dict)
        sampling_results.append(hopk_sampling)

    return sampling_results
