from script.dataset import Dataset
from script.prepare import prepare
from script.utils import load_config
from script.pipeline import Pipeline


def train_and_test(data, dataset_root, config):
    """模型训练和测试

        使用给定数据和配置训练并测试模型

        Inputs:
        -------
        data: string, 使用的数据集名称, ['cora', 'pubmed', 'citeseer']
        dataset_root: string, 数据集保存根文件夹路径
        config: dict, 参数配置

    """

    # 数据获取和预处理
    dataset = Dataset(data, dataset_root)
    prep_dataset = prepare(dataset)

    # 训练模型
    pipeline = Pipeline(**config[data])
    pipeline.train(prep_dataset)

    # 测试集准确率
    test_acc = pipeline.predict(prep_dataset, 'test')
    print('[{}] Test Accuracy: {:.3f}\n'.format(data.upper(), test_acc))

    return


if __name__ == '__main__':

    # 数据集根目录
    dataset_root = '../../Dataset'

    # 加载全局配置
    config = load_config(config_file='config.yaml')

    # 使用Cora数据集训练和测试模型
    train_and_test('cora', dataset_root, config)
    # Cora Test Accuracy: 0.819

    # 使用Citeseer数据集训练和测试模型
    train_and_test('citeseer', dataset_root, config)
    # Citeseer Test Accuracy: 0.702

    # 使用Pubmed数据集训练和测试模型
    train_and_test('pubmed', dataset_root, config)
    # Pubmed Test Accuracy: 0.790
