"""
"""


import torch
import torch.nn as nn
import torch.optim as optim

from .model import GCNet


class Pipeline(object):

    def __init__(self, **params):
        """
        """

        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])

        return

    def __build_model(self, **model_params):
        """
        """

        self.model = GCNet(**model_params)
        if torch.cuda.is_available():
            self.model.cuda()

        return

    def __build_components(self, **hyper_params):
        """
        """

        self.epochs = hyper_params['epochs']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['weight_decay']
        )

        return

    def train(self, dataset):
        """
        """

        train_y = dataset.y[dataset.train_mask]
        for epoch in range(self.epochs):
            self.model.train()

            logits = self.model(dataset.adjacency, dataset.X)
            train_logits = logits[dataset.train_mask]
            loss = self.criterion(train_logits, train_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_acc = self.predict(dataset, 'train')
            valid_acc = self.predict(dataset, 'valid')

            print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]-[ValidAcc:{:.4f}]'.format(
                epoch, loss, train_acc, valid_acc))

        return

    def predict(self, dataset, split='train'):
        """
        """

        self.model.eval()

        if split == 'train':
            mask = dataset.train_mask
        elif split == 'valid':
            mask = dataset.valid_mask
        else:  # split == 'test'
            mask = dataset.test_mask

        logits = self.model(dataset.adjacency, dataset.X)
        predict_y = logits[mask].max(1)[1]

        y = dataset.y[mask]
        accuracy = torch.eq(predict_y, y).float().mean()

        return accuracy
