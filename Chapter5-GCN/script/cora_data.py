"""Cora数据集预处理

"""


import os
import time
import pickle
import urllib.request

from collections import namedtuple


Data = namedtuple(
    typename='Data',
    field_names=[
        'x', 'y', 'adjacency', 'train_mask',
        'val_mask', 'test_mask'
    ]
)


class CoraData(object):

    download_url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    filenames = ['ind.cora.{}'.format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root, rebuild=False):
        """
        """

        self.data_root = data_root
        save_file = os.path.join(data_root, 'processed_cora.pkl')

        if os.path.isfile(save_file) and not rebuild:
            print('Using Cached File: {}'.format(save_file))
            self._data = pickle.load(open(save_file, 'rb'))
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, 'wb') as f:
                pickle.dump(self.data, f)
            print('Cached File: {}'.format(save_file))

        return

    @property
    def data(self):
        return self._data

    def maybe_download(self):
        """
        """

        save_path = os.path.join(self.data_root, 'raw')
        for name in self.filenames:
            if not os.path.isfile(os.path.join(save_path, name)):
                self.download_data('{}/{}'.format(self.download_url, name), save_path)
                time.sleep(2)

        return

    def download_data(self, url, save_path):
        """
        """

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        try:
            data = urllib.request.urlopen(url, timeout=100)
            filename = os.path.basename(url)

            with open(os.path.join(save_path, filename), 'wb') as f:
                f.write(data.read())

            data.close()
        except Exception:
            self.download_data(url, save_path)

        return


if __name__ == '__main__':

    cora_data = CoraData(data_root='../data/cora', rebuild=False)
