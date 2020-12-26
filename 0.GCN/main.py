from script.cora import CoraData
from script.pipeline import Pipeline
from script.prepare import prepare_data


if __name__ == '__main__':

    # 数据下载与预处理
    cora = CoraData(output_dir='./data/cora', rebuild=False)
    dataset = prepare_data(cora)

    # 训练参数
    params = {
        'random_state': 42,       # 随机种子
        'model': {
            'input_dim': 1433,    # 节点特征维度
            'output_dim': 7,      # 节点类别数
            'hidden_dim': 16,     # 隐层输出特征维度
            'use_bias': True      # 是否使用偏置
        },
        'hyper': {
            'lr': 1e-2,           # 优化器初始学习率
            'epochs': 200,        # 训练轮次
            'weight_decay': 5e-4  # 优化器权重衰减
        }
    }

    # 训练模型
    pipeline = Pipeline(**params)
    pipeline.train(dataset)

    # 测试集准确率
    test_acc = pipeline.predict(dataset, 'test')
    print('Test Accuracy: {:.6f}'.format(test_acc))
    # Test Accuracy: 0.819
