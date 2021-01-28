"""加载图数据

    使用TUDataset下载并与处理相关数据集
    预处理后数据集保存路径: $dataset_root/$data

"""


import random

from torch_geometric.datasets import TUDataset


class Dataset(object):
    """加载图数据

        使用TUDataset下载并与处理相关数据集

    """

    def __init__(self, data, dataset_root, **params):
        """加载图数据

            使用TUDataset下载并与处理相关数据集, TUDataset文档见
            https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.TUDataset

            预处理后数据集保存路径: $dataset_root/$data

            Inputs:
            -------
            data: string, 使用的数据集名称, ['DD', 'NCI1', 'PROTEINS']
            dataset_root: string, 数据集保存根文件夹路径
            params: dict, 包含split子字典, 提供验证集和测试集数据比例

        """

        assert data in ['DD', 'NCI1', 'PROTEINS'], 'uknown dataset'

        # 下载并预处理相关数据集
        print('Downloading and Preprocessing [{}] Dataset ...'.format(data.upper()))
        self.dataset = TUDataset(
            root=dataset_root, name=data, use_node_attr=True
        )

        # 数据集划分
        self.__split_data(**params)

        return

    def __split_data(self, **params):
        """数据集划分

            将数据集划分为训练集、验证集和测试集

            Input:
            ------
            params: dict, 包含split子字典, 提供验证集和测试集数据比例

        """

        # 图数量
        num_graphs = len(self.dataset)

        # 打乱初始图样本索引
        indices = list(range(num_graphs))
        random.seed(params['random_state'])
        random.shuffle(indices)

        # 测试集和验证集图数量
        num_test = int(num_graphs * params['split']['test_prop'])
        num_valid = int(num_graphs * params['split']['valid_prop'])

        # 获得测试集、验证集和训练集图样本索引
        test_indices = indices[:num_test]
        valid_indices = indices[num_test:num_test + num_valid]
        train_indices = indices[num_test + num_valid:]

        # 获取测试集、验证集和训练集图样本
        self.test = self.dataset[test_indices]
        self.valid = self.dataset[valid_indices]
        self.train = self.dataset[train_indices]
        # 获得训练集中所有图所包含的平均节点数量
        self.avg_nodes = int(self.train.data.x.size(0) / len(self.train))

        # 每一组数据使用方法, 以self.test为例
        # 使用Dataloader加载batch
        # from torch_geometric.data import DataLoader
        # test_loader = DataLoader(
        #     dataset=self.test, batch_size=1, shuffle=False
        # )
        #
        # 遍历每个batch
        # for i, data in enumerate(test_loader):
        #     X = data.x                    # batch中所有图的节点特征
        #     edge_index = data.edge_index  # batch中所有图的边列表
        #     batch = data.batch            # batch中指明节点属于哪个图
        #     y = data.y                    # batch中所有图的标签

        return
