"""加载图数据
"""


import torch

from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset


class Indegree(object):
    """
    """

    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        col, x = data.edge_index[1], data.x
        deg = degree(col, data.num_nodes)

        if self.norm:
            deg = deg / (deg.max() if self.max is None else self.max)

        deg = deg.view(-1, 1)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data


class Dataset(object):
    """
    """

    def __init__(self, data, dataset_root):
        """
        """

        assert data in ['DD', 'NCI1', 'PROTEINS'], 'uknown dataset'

        print('Downloading and Preprocessing [{}] Dataset ...'.format(data))

        dataset = TUDataset(
            root=dataset_root, name=data,
            pre_transform=Indegree(), use_node_attr=True
        )

        self.data = dataset.data
        self.slices = dataset.slices

        return


if __name__ == '__main__':

    # data = 'DD'
    # # data = 'NCI1'
    # # data = 'PROTEINS'

    # dataset_root = '../../../Dataset'

    # dataset_name = data.upper()
    # dataset_dir = os.path.join(dataset_root, data)
    # print(dataset_dir)

    # dataset = TUDataset(
    #     root=dataset_root, name=data,
    #     pre_transform=Indegree(), use_node_attr=True
    # )

    # print(dataset.data.x.size())
    # print(dataset.data.y.size())
    # print(dataset.data.edge_index.size())
    # print(dataset.slices['x'].size())
    # print(dataset.slices['y'].size())
    # print(dataset.slices['edge_index'].size())

    # dataset = Dataset(data='DD', dataset_root='../../../Dataset')
    # dataset = Dataset(data='NCI1', dataset_root='../../../Dataset')
    dataset = Dataset(data='PROTEINS', dataset_root='../../../Dataset')

    print(dataset.data.x.size())
    print(dataset.data.y.size())
    print(dataset.data.edge_index.size())
    print(dataset.slices['x'].size())
    print(dataset.slices['y'].size())
    print(dataset.slices['edge_index'].size())
