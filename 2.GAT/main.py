from script.cora import CoraData
# from script.pipeline import Pipeline
from script.prepare import prepare_data


if __name__ == '__main__':

    # 数据下载与预处理
    cora = CoraData(output_dir='./data/cora', rebuild=False)
    dataset = prepare_data(cora, sparse=False)

    # 训练参数
    params = {
        'model': {
            'input_dim': 1433,  # 节点特征维度
            'hidden_dim': 8,    # 隐层输出特征维度
            'output_dim': 7,    # 节点类别个数
            'num_heads': 8,     # 多头注意力个数
            'dropout': 0.6,     # dropout比例
            'alpha': 0.2        # LeakyReLU斜率
        },
        'hyper': {
            'lr': 3e-3,           # 优化器初始学习率
            'epochs': 10,         # 训练轮次
            'patience': 100,      # 早停轮次
            'weight_decay': 5e-4  # 优化器权重衰减
        }
    }

    # # 训练模型
    # pipeline = Pipeline(**params)
    # pipeline.train(dataset)

    # # 测试集准确率
    # test_acc = pipeline.predict(dataset, 'test')
    # print('Test Accuracy: {:.6f}'.format(test_acc))
