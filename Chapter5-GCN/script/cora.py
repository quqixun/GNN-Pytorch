"""Cora数据集预处理

    Cora数据集的下载、存储与预处理,
    原始数据集存储路径: $output_dir/raw
    预处理数据集存储路径: $output_dir/processed_cora.pkl

"""


import os
import time
import pickle
import urllib.request

from utils import *
from collections import namedtuple


Data = namedtuple(
    typename='Data',
    field_names=[
        'x', 'y', 'adjacency', 'train_mask',
        'val_mask', 'test_mask'
    ]
)


class CoraData(object):
    """Cora数据集的下载、存储与预处理

        原始Cora数据集的下载、存储与预处理,
        原始数据集存储路径: $output_dir/raw
        预处理数据集存储路径: $output_dir/processed_cora.pkl

    """

    url_root = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    filenames = [
        'ind.cora.x', 'ind.cora.tx', 'ind.cora.allx',
        'ind.cora.y', 'ind.cora.ty', 'ind.cora.ally',
        'ind.cora.graph', 'ind.cora.test.index'
    ]

    def __init__(self, output_dir, rebuild=False):
        """CoraData初始化

            原始Cora数据集的下载、存储与预处理,
            原始数据集存储路径: $output_dir/raw
            预处理数据集存储路径: $output_dir/processed_cora.pkl

            Inputs:
            -------
            output_dir: string, 数据集保存文件夹路径
            rebuild: boolean, 是否重新生成Cora数据集

        """

        print(TITLE_STRING.format('DATASET PREPROCESSING'))

        self.raw_dir = os.path.join(output_dir, 'raw')
        self.prep_file = os.path.join(output_dir, 'processed_cora.pkl')

        if os.path.isfile(self.prep_file) and not rebuild:
            # 预处理Cora数据集已存在且不更新
            print('[Step:1/1] Loading Cached File:', self.prep_file)
            self.data = pickle.load(open(self.prep_file, 'rb'))
        else:
            # 预处理Cora数据集不存在或更新现有数据集
            print('[Step:1/3] Downloading Raw Cora Dataset ...')
            self.download_data()

            # print('[Step:2/3] Processing Cora Dataset ...')
            # self.data = self.process_data()

            # print('[Step:3/3] Cached File:', self.prep_file)
            # with open(self.prep_file, 'wb') as f:
            #     pickle.dump(self.data, f)

        return

    # ------------------------------------------------------------------------
    # 下载Cora数据集

    def download_data(self):
        """下载原始Cora数据集

            分别下载各数据文件, 保存至self.raw_dir文件夹

        """

        # 生成self.raw_dir文件夹
        create_dir(self.raw_dir)

        for name in self.filenames:
            # 遍历各数据文件
            file = os.path.join(self.raw_dir, name)
            if not os.path.isfile(file):
                # 若数据文件不存在则下载文件
                url = '{}/{}'.format(self.url_root, name)
                self.download_from_url(url, file)

        return

    def download_from_url(self, url, file):
        """根据url下载文件

            从url下载文件保存至file

            Inputs:
            -------
            url: string, 文件下载链接
            file: string, 下载文件保存路径

        """

        try:
            # 建立文件连接, 写入文件
            data = urllib.request.urlopen(url, timeout=100)
            with open(file, 'wb') as f:
                f.write(data.read())
            data.close()
        except Exception:
            # 下载失败, 尝试再次下载
            self.download_from_url(url, file)

        return

    # ------------------------------------------------------------------------
    # 预处理Cora数据集

    def process_data(self):
        """
        """

        return


if __name__ == '__main__':

    cora_data = CoraData(output_dir='../data/cora', rebuild=False)
