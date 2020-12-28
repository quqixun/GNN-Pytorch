# GNN-Pytorch
GNN方法和模型的Pytorch实现。

<br/>

## 归纳学习

| No.  |            方法            |        状态        |                     论文                      | Cora  | Pubmed | Citeseer |
| :--: | :------------------------: | :----------------: | :-------------------------------------------: | :---: | :----: | :------: |
|  0   |       [GCN](./0.GCN)       | :heavy_check_mark: | [Paper](https://arxiv.org/pdf/1609.02907.pdf) | 0.819 | 0.790  |  0.702   |
|  1   | [GraphSAGE](./1.GraphSAGE) | :heavy_check_mark: | [Paper](https://arxiv.org/pdf/1706.02216.pdf) | 0.801 | 0.778  |  0.701   |
|  2   |       [GAT](./2.GAT)       | :heavy_check_mark: | [Paper](https://arxiv.org/pdf/1710.10903.pdf) |       |        |          |

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