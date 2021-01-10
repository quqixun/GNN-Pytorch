# GNN-Pytorch
GNN方法和模型的Pytorch实现。Pytorch implementation of GNN.

<br/>

## 节点分类 - Node Classification

使用的数据集列表：

|    Dataset    | Nodes | Edges | Features | Classes | Train | Valid | Test |
| :-----------: | :---: | :---: | :------: | :-----: | :---: | :---: | :--: |
|     Cora      | 2708  | 5429  |   1433   |    7    |  140  |  500  | 1000 |
|   Cora-Full   | 2708  | 5429  |   1433   |    7    | 1208  |  500  | 1000 |
|   Citeseer    | 3327  | 4732  |   3703   |    6    |  120  |  500  | 1000 |
| Citeseer-Full | 3327  | 4732  |   3703   |    6    | 1827  |  500  | 1000 |
|    Pubmed     | 19717 | 44338 |   500    |    3    |  60   |  500  | 1000 |
|  Pubmed-Full  | 19717 | 44338 |   500    |    3    | 18217 |  500  | 1000 |

各方法实验结果列表：

|        Status        |             Method        |                             Paper                             | Cora  | Pubmed | Citeseer |
| :----------------: | :---------------------------: | :----------------------------------------------------------: | :---: | :----: | :------: |
| :heavy_check_mark: |       [GCN](./Node/GCN)       | [Kipf and Welling, 2017](https://arxiv.org/pdf/1609.02907.pdf) | 0.819 | 0.790  |  0.702   |
| :heavy_check_mark: | [GraphSAGE](./Node/GraphSAGE) | [Hamilton et al., 2017](https://arxiv.org/pdf/1706.02216.pdf) | 0.801 | 0.778  |  0.701   |
| :heavy_check_mark: |       [GAT](./Node/GAT)       | [Velickovic et al., 2018](https://arxiv.org/pdf/1710.10903.pdf) | 0.823 |  0.777   |  0.715   |
| :heavy_check_mark: | [FastGCN](./Node/FastGCN)<sup>**\***</sup> | [Chen et al., 2018](https://arxiv.org/pdf/1801.10247.pdf) | 0.854 | 0.858 | 0.779 |

**\*** 使用Cora-Full，Pubmed-Full和Citeseer-Full数据集训练并评价。

<br/>

## 图分类 - Graph Classification

<br/>

## 图表示学习 - Graph Representing Learning

<br>

## 环境配置 - Packages

| 依赖    | 版本   | 安装                                                         |
| ------- | ------ | ------------------------------------------------------------ |
| python  | 3.8.6  | conda create --name gnn python=3.8.6                         |
| numpy   | 1.19.4 | pip install numpy==1.19.4                                    |
| scipy   | 1.5.4  | pip install scipy==1.5.4                                     |
| PyYAML  | 5.3.1  | pip install PyYAML==5.3.1                                    |
| pytorch | 1.7.0  | cpu：conda install pytorch\==1.7.0 cpuonly -c pytorch<br/>gpu：conda install pytorch\==1.7.0 cudatoolkit=10.2 -c pytorch |