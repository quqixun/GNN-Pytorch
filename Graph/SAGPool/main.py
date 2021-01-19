from script.dataset import Dataset
# from script.prepare import prepare
from script.utils import load_config
from script.pipeline import Pipeline


def train_and_test(model_name, data, dataset_root, config):
    """模型训练和测试

        使用给定数据和配置训练并测试模型

        Inputs:
        -------
        data: string, 使用的数据集名称, ['DD', 'NCI1', 'PROTEINS']
        dataset_root: string, 数据集保存根文件夹路径
        config: dict, 参数配置

    """

    # 数据获取和预处理
    # print(config[data])
    dataset = Dataset(data, dataset_root, **config[data])

    # 训练模型
    pipeline = Pipeline(model_name, **config[data])
    pipeline.train(dataset)

    # 测试集准确率
    test_loss, test_f1 = pipeline.predict(dataset, 'test')
    print('[{}]-[TestLoss:{:.4f}]-[TestF1:{:.3f}]\n'.format(
        data.upper(), test_loss, test_f1))

    return


if __name__ == '__main__':

    # 数据集根目录
    dataset_root = '../../Dataset'

    # 加载全局配置
    config = load_config(config_file='config.yaml')

    # 使用DD数据集训练和测试模型
    # train_and_test('DD', dataset_root, config)
    # [SAGPoolG] DD Test F1 Score:
    # [SAGPoolH] DD Test F1 Score:

    # 使用NCI1数据集训练和测试模型
    # train_and_test('SAGPoolG', 'NCI1', dataset_root, config)
    # [SAGPoolG] NCI1 Test F1 Score: 0.748
    train_and_test('SAGPoolH', 'NCI1', dataset_root, config)
    # [SAGPoolH] NCI1 Test F1 Score:

    # 使用PROTEINS数据集训练和测试模型
    # train_and_test('PROTEINS', dataset_root, config)
    # [SAGPoolG] PROTEINS Test F1 Score:
    # [SAGPoolH] PROTEINS Test F1 Score:
