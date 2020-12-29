# GNN-Pytorch
GNN方法和模型的Pytorch实现。

<br/>

## 节点分类

|        状态        |             方法              |                     论文                      | Cora  | Pubmed | Citeseer |
| :----------------: | :---------------------------: | :-------------------------------------------: | :---: | :----: | :------: |
| :heavy_check_mark: |       [GCN](./Node/GCN)       | [Paper](https://arxiv.org/pdf/1609.02907.pdf) | 0.819 | 0.790  |  0.702   |
| :heavy_check_mark: | [GraphSAGE](./Node/GraphSAGE) | [Paper](https://arxiv.org/pdf/1706.02216.pdf) | 0.801 | 0.778  |  0.701   |
| :heavy_check_mark: |       [GAT](./Node/GAT)       | [Paper](https://arxiv.org/pdf/1710.10903.pdf) |       |        |          |

<br/>

## 图池化

<br/>

## 环境配置

| 依赖    | 版本   | 安装                                                         |
| ------- | ------ | ------------------------------------------------------------ |
| python  | 3.8.6  | conda create --name gnn python=3.8.6                         |
| numpy   | 1.19.4 | pip install numpy==1.19.4                                    |
| scipy   | 1.5.4  | pip install scipy==1.5.4                                     |
| PyYAML  | 5.3.1  | pip install PyYAML==5.3.1                                    |
| pytorch | 1.7.0  | cpu：conda install pytorch\==1.7.0 cpuonly -c pytorch<br/>gpu：conda install pytorch\==1.7.0 cudatoolkit=10.2 -c pytorch |