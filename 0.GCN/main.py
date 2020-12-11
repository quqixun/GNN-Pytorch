from script.cora import CoraData
from script.pipeline import Pipeline
from script.prepare import prepare_data


if __name__ == '__main__':

    # 数据准备
    cora = CoraData(output_dir='./data/cora', rebuild=False)
    dataset = prepare_data(cora)

    # 训练参数
    params = {
        'model': {
            'input_dim': 1433,
            'output_dim': 7,
            'hidden_dim': 16,
            'use_bias': True
        },
        'hyper': {
            'lr': 1e-2,
            'epochs': 100,
            'weight_decay': 5e-4
        }
    }

    # 训练模型
    pipeline = Pipeline(**params)
    pipeline.train(dataset)

    # 测试集准确率
    test_acc = pipeline.predict(dataset, 'test')
    print('Test Accuracy: {:.6f}'.format(test_acc))
