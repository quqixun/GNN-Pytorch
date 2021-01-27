from script.dataset import Dataset
from script.utils import load_config
from script.pipeline import Pipeline


def train_and_test(data, dataset_root, config):
    """模型训练和测试

        使用给定数据和配置训练并测试模型

        Inputs:
        -------
        data: string, 使用的数据集名称, ['DD', 'NCI1', 'PROTEINS']
        dataset_root: string, 数据集保存根文件夹路径
        config: dict, 参数配置

    """

    # 数据获取和预处理
    dataset = Dataset(data, dataset_root, **config[data])

    # 训练模型
    pipeline = Pipeline(**config[data])
    pipeline.train(dataset)

    # 测试集准确率
    # test_loss, test_acc = pipeline.predict(dataset, 'test')
    # print('[{}]-[TestLoss:{:.4f}]-[TestAcc:{:.3f}]\n'.format(
    #     data.upper(), test_loss, test_acc))

    return


if __name__ == '__main__':

    # 数据集根目录
    dataset_root = '../../Dataset'

    # 加载全局配置
    config = load_config(config_file='config.yaml')

    # 使用DD数据集训练和测试模型
    # train_and_test('DD', dataset_root, config)
    # DD Test Accuracy:

    # 使用NCI1数据集训练和测试模型
    train_and_test('NCI1', dataset_root, config)
    # NCI1 Test Accuracy:

    # 使用PROTEINS数据集训练和测试模型
    # train_and_test('PROTEINS', dataset_root, config)
    # PROTEINS Test Accuracy:
