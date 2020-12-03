"""辅助函数及全局变量
"""


__all__ = ['TITLE_STRING', 'create_dir']


import os


# 每一步骤Title
TITLE_STRING = '\n' + '-' * 75 + '\n{}\n'


def create_dir(dir_path):
    # 生成文件夹
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return
