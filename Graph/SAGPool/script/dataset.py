"""加载图数据
"""


import random

from torch_geometric.datasets import TUDataset


class Dataset(object):
    """
    """

    def __init__(self, data, dataset_root, **params):
        """
        """

        assert data in ['DD', 'NCI1', 'PROTEINS'], 'uknown dataset'

        print('Downloading and Preprocessing [{}] Dataset ...'.format(data.upper()))
        self.__load_data(data, dataset_root)
        self.__split_data(**params)

        return

    def __load_data(self, data, dataset_root):
        """
        """

        self.dataset = TUDataset(
            root=dataset_root, name=data, use_node_attr=True
        )

        return

    def __split_data(self, **params):
        """
        """

        num_graphs = len(self.dataset)
        num_test = int(num_graphs * params['split']['test_prop'])
        num_valid = int(num_graphs * params['split']['valid_prop'])

        indices = list(range(num_graphs))
        random.seed(params['random_state'])
        random.shuffle(indices)

        test_indices = indices[:num_test]
        valid_indices = indices[num_test:num_test + num_valid]
        train_indices = indices[num_test + num_valid:]

        self.test = self.dataset[test_indices]
        self.valid = self.dataset[valid_indices]
        self.train = self.dataset[train_indices]

        return
