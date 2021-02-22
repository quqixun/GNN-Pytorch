# GNN-Pytorch
GNN方法和模型的Pytorch实现。Pytorch implementation of GNN.

这里更注重方法的学习，而不是获得更高的结果。

<br/>

## 节点分类 - Node Classification

使用的数据集列表，有条件可以使用[OGB数据集](https://github.com/snap-stanford/ogb)：

|    Dataset    | Nodes | Edges | Node Attr. | Classes | Train | Valid | Test |
| :-----------: | :---: | :---: | :--------: | :-----: | :---: | :---: | :--: |
|     Cora      | 2708  | 5429  |    1433    |    7    |  140  |  500  | 1000 |
|   Cora-Full   | 2708  | 5429  |    1433    |    7    | 1208  |  500  | 1000 |
|   Citeseer    | 3327  | 4732  |    3703    |    6    |  120  |  500  | 1000 |
| Citeseer-Full | 3327  | 4732  |    3703    |    6    | 1827  |  500  | 1000 |
|    Pubmed     | 19717 | 44338 |    500     |    3    |  60   |  500  | 1000 |
|  Pubmed-Full  | 19717 | 44338 |    500     |    3    | 18217 |  500  | 1000 |

各方法实验结果(Accuracy)列表：

|        Status        |             Method        |                             Paper                             | Cora  | Citeseer | Pubmed |
| :----------------: | :---------------------------: | :----------------------------------------------------------: | :---: | :------: | :------: |
| :heavy_check_mark: |       [GCN](./Node/GCN)       | [Kipf and Welling, 2017](https://arxiv.org/pdf/1609.02907.pdf) | 0.819 |  0.702   | 0.790 |
| :heavy_check_mark: | [GraphSAGE](./Node/GraphSAGE) | [Hamilton and Ying et al., 2017](https://arxiv.org/pdf/1706.02216.pdf) | 0.801 |  0.701   | 0.778 |
| :heavy_check_mark: |       [GAT](./Node/GAT)       | [Velickovic et al., 2018](https://arxiv.org/pdf/1710.10903.pdf) | 0.824 |  0.719  | 0.782 |
| :heavy_check_mark: | [FastGCN](./Node/FastGCN)<sup>**\***</sup> | [Chen and Ma et al., 2018](https://arxiv.org/pdf/1801.10247.pdf) | 0.854 | 0.779 | 0.855 |
| :heavy_check_mark: | [GRAND](./Node/GRAND) | [Feng and Zhang et al., 2020](https://arxiv.org/pdf/2005.11079.pdf) | 0.839 | 0.726 | 0.797 |

**\*** 使用Cora-Full，Pubmed-Full和Citeseer-Full数据集训练并评价。

<br/>

## 图分类 - Graph Classification

使用的数据集列表，更多的数据集见[TUDataset](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)，有条件可以使用[OGB数据集](https://github.com/snap-stanford/ogb)：

| Dataset  | Graphs | Avg. Nodes | Avg. Edges | Node Attr. | Classes | Train | Valid | Test |
| :------: | :----: | :--------: | :--------: | :--------: | :-----: | :---: | :---: | :--: |
|    DD    |  1178  |   284.32   |   715.66   |     89     |    2    |  826  |  117  | 235  |
|   NCI1   |  4110  |   29.87    |   32.30    |     37     |    2    | 2877  |  411  | 822  |
| PROTEINS |  1113  |   39.06    |   72.82    |     4      |    2    |  780  |  111  | 222  |

各方法实验结果(Accuracy)列表：

|       Status       |                 Method                 |                            Paper                             |  DD   | NCI1  | PROTEINS |
| :----------------: | :------------------------------------: | :----------------------------------------------------------: | :---: | :---: | :------: |
|                    |                DiffPool                |  [Ying et al., 2018](https://arxiv.org/pdf/1806.08804.pdf)   |       |       |          |
|                    |   [Graph U-Nets](./Graph/GraphUNets)   |   [Gao et al., 2019](https://arxiv.org/pdf/1905.05178.pdf)   |       |       |          |
| :heavy_check_mark: | [SAGPool<sub>g</sub>](./Graph/SAGPool) | [Lee and Lee et al., 2019](https://arxiv.org/pdf/1904.08082.pdf) | 0.723 | 0.763 |  0.757   |
| :heavy_check_mark: | [SAGPool<sub>h</sub>](./Graph/SAGPool) | [Lee and Lee et al., 2019](https://arxiv.org/pdf/1904.08082.pdf) | 0.745 | 0.648 |  0.743   |
| :heavy_check_mark: |    [MinCutPool](./Graph/MinCutPool)    | [Bianchi and Grattarola et al., 2020](https://arxiv.org/pdf/1907.00481.pdf) | 0.770 | 0.742 |  0.766   |

<br/>

## 环境配置 - Packages

| 依赖            | 版本   | 安装                                                         |
| --------------- | ------ | ------------------------------------------------------------ |
| python          | 3.8.6  | conda create --name gnn python=3.8.6                         |
| numpy           | 1.20.0 | pip install numpy==1.20.0                                    |
| scipy           | 1.6.0  | pip install scipy==1.6.0                                     |
| pyyaml          | 5.4.1  | pip install pyyaml==5.4.1                                    |
| scikit-learn    | 0.24.1 | pip install scikit-learn==0.24.1                             |
| pytorch         | 1.7.1  | conda install pytorch\==1.7.1 cudatoolkit=11.0 -c pytorch    |
| torch-geometric | 1.6.3  | [Installation](https://github.com/rusty1s/pytorch_geometric#installation) |

