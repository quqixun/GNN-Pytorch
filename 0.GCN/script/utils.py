"""辅助函数及全局变量
"""


import os


def create_dir(dir_path):
    # 生成文件夹
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return
