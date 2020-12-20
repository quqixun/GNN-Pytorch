from script.cora import CoraData
from script.pipeline import Pipeline
from script.prepare import prepare_data


if __name__ == '__main__':

    # 数据下载与预处理
    cora = CoraData(output_dir='./data/cora', rebuild=False)
    dataset = prepare_data(cora)

    # # 训练参数
    # params = {
    #     'model': {
    #         'input_dim': 1433,              # 节点特征维度
    #         'hidden_dims': [128, 7],        # 隐层输出特征维度
    #         'num_neighbors_list': [10, 10]  # 没接采样邻居的节点数
    #     },
    #     'hyper': {
    #         'lr': 3e-3,                # 优化器初始学习率
    #         'epochs': 10,              # 训练轮次
    #         'batch_size': 16,          # 批数据大小
    #         'weight_decay': 5e-4,      # 优化器权重衰减
    #         'num_batch_per_epoch': 20  # 每个epoch循环的批次数
    #     }
    # }

    # # 训练模型
    # pipeline = Pipeline(**params)
    # pipeline.train(dataset)

    # # 测试集准确率
    # test_acc = pipeline.predict(dataset, 'test')
    # print('Test Accuracy: {:.6f}'.format(test_acc))
