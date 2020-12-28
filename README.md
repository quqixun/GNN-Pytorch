# GNN-Pytorch
<br/>

一些图神经网络的实现。

- [x] [0. GCN](./0.GCN) - [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)
- [x] [1. GraphSAGE](./1.GraphSAGE) - [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)
- [ ] [2. GAT](./2.GAT)

<br/>

## 环境配置

| 依赖    | 版本   | 安装                                                         |
| ------- | ------ | ------------------------------------------------------------ |
| python  | 3.8.6  | conda create --name gnn python=3.8.6                         |
| numpy   | 1.19.4 | pip install numpy==1.19.4                                    |
| scipy   | 1.5.4  | pip install scipy==1.5.4                                     |
| PyYAML  | 5.3.1  | pip install PyYAML==5.3.1                                    |
| pytorch | 1.7.0  | cpu：conda install pytorch\==1.7.0 cpuonly -c pytorch<br/>gpu：conda install pytorch\==1.7.0 cudatoolkit=10.2 -c pytorch |