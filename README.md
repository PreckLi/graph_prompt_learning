# Graph Prompt Learning
This is the official implementation of Graph Prompt Learning
## Requirements
torch==1.10.0
torch_geometric==2.0.0
## Datasets
The node classification datasets include Cora,Citeseer,Pubmed,Computers,CS,Photo,Physics and DBLP.
The graph classification datasets include DD，PROTEINS,NCI1 and NCI109.
Specificially,the download url of datasets can refer to https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
## Code
For node classification, the code is in [Node_Classification_GPL](https://github.com/PreckLi/graph_prompt_learning/tree/main/Node_Classification_GPL), and the code for GCN and GAT on the Cora dataset according to the original division method is in [Py_GCN_GPL](https://github.com/PreckLi/graph_prompt_learning/tree/main/Py_GCN_GPL/pygcn) and [Py_GAT_GPL](https://github.com/PreckLi/graph_prompt_learning/tree/main/Py_GAT_GPL).
For graph classification, the code is in [Graph_Classification_GPL](https://github.com/PreckLi/graph_prompt_learning/tree/main/Graph_Classification_GPL), [DiffPool_GPL](https://github.com/PreckLi/graph_prompt_learning/tree/main/Diffpool_GPL), SAGPool and [Mewispool_GPL](https://github.com/PreckLi/graph_prompt_learning/tree/main/Mewispool_GPL/graph_classification) are implemented separately.
