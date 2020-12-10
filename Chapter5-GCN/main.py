from script.cora import CoraData
from script.prepare import prepare_data
from script.pipeline import Trainer


if __name__ == '__main__':

    # 数据准备
    cora = CoraData(output_dir='./data/cora', rebuild=False)
    dataset = prepare_data(cora)

    #
    params = {
        'model': {
            'input_dim': 1433,
            'output_dim': 7,
            'hidden_dim': 16,
            'use_bias': True
        },
        'hyper': {
            'lr': 1e-3,
            'weight_decay': 5e-4
        }
    }

    trainer = Trainer(**params)
