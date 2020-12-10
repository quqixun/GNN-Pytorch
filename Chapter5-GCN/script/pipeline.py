"""
"""


import torch
import torch.nn as nn
import torch.optim as optim

from .model import GCNet


class Trainer(object):

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
        self.model.train()
        if torch.cuda.is_available():
            self.model.cuda()

        return

    def __build_components(self, **hyper_params):
        """
        """

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            **hyper_params
        )

        return
