import torch
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, LGConv, TransformerConv, ARMAConv, ChebConv, GeneralConv, \
    SuperGATConv
from torch import nn
from torch.nn import functional as F
from layers import Degree_layer, GraphAttentionLayer
import os


class GNN(nn.Module):
    def __init__(self, data: torch_geometric.data.data.Data, hid_size, gnn_style='gcn', dropout=0.25, alpha=0.2):
        super().__init__()
        self.hid_size = hid_size
        self.dropout = dropout
        self.alpha = alpha
        if gnn_style == 'gcn':
            self.conv1 = GCNConv(data.num_node_features, hid_size)
            self.conv2 = GCNConv(hid_size, hid_size)
        if gnn_style == 'gat':
            self.conv1 = GATConv(data.num_node_features, hid_size, heads=8)
            self.conv2 = GATConv(hid_size * 8, hid_size, heads=1)
        if gnn_style == 'graphsage':
            self.conv1 = SAGEConv(data.num_node_features, hid_size)
            self.conv2 = SAGEConv(hid_size, hid_size)
        if gnn_style == 'transformerconv':
            self.conv1 = TransformerConv(data.num_node_features, hid_size, heads=4)
            self.conv2 = TransformerConv(hid_size * 4, hid_size, heads=1)
        if gnn_style == 'lgconv':
            self.conv1 = LGConv()
            self.lin1 = nn.Linear(data.num_node_features, hid_size)
            self.conv2 = LGConv()
            self.lin2 = nn.Linear(hid_size, data.num_classes)

            self.lg_linear1 = nn.Linear(hid_size, 1)
            self.lg_linear2 = nn.Linear(hid_size, 1)
        if gnn_style == 'ARMAconv':
            self.conv1 = ARMAConv(data.num_node_features, hid_size)
            self.conv2 = ARMAConv(hid_size, hid_size)
        if gnn_style == "Chebconv":
            self.conv1 = ChebConv(data.num_node_features, hid_size, K=3)
            self.conv2 = ChebConv(hid_size, hid_size, K=3)
        if gnn_style == "Generalconv":
            self.conv1 = GeneralConv(data.num_node_features, hid_size)
            self.conv2 = GeneralConv(hid_size, hid_size)
        if gnn_style == "Supergatconv":
            self.conv1 = SuperGATConv(data.num_node_features, hid_size, heads=8)
            self.conv2 = SuperGATConv(hid_size * 8, hid_size, heads=1)
        # node classification linear
        self.linear1 = nn.Linear(hid_size, data.num_classes)
        # node degree prediction linear
        self.linear2 = Degree_layer(hid_size, 1)
        # node neighbors' degree distribution
        self.linear3 = nn.Linear(hid_size, data.degree_exist_matrix.shape[1])

        self.down_sample = nn.Linear(data.num_node_features, hid_size)

    def forward(self, data: torch_geometric.data.data.Data, gnn_style):
        x, edge_index = data.x, data.edge_index
        if gnn_style == 'gcn':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x

            result1 = F.log_softmax(x, -1)  # <batch, N, F >
            result2 = self.linear2(x)
            result3 = self.linear3(x)

        if gnn_style == 'gat':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x

            result1 = F.log_softmax(x, -1)  # <batch, N, F >
            result2 = self.linear2(x)
            result3 = self.linear3(x)

        if gnn_style == 'graphsage':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x

            result1 = F.log_softmax(x, -1)  # <batch, N, F >
            result2 = self.linear2(x)
            result3 = self.linear3(x)
        if gnn_style == 'transformerconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x

            result1 = F.log_softmax(x, 1)
            result2 = self.linear2(x)
            result3 = self.linear3(x)
        if gnn_style == 'lgconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            classify_x = self.conv2(x, edge_index)
            x = classify_x + original_x

            result1 = F.log_softmax(self.lin2(x), 1)
            result2 = self.lg_linear1(x)
            result3 = self.lg_linear2(x)
        if gnn_style == 'ARMAconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x

            result1 = F.log_softmax(x, -1)  # <batch, N, F >
            result2 = self.linear2(x)
            result3 = self.linear3(x)
        if gnn_style == 'Chebconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x

            result1 = F.log_softmax(x, -1)  # <batch, N, F >
            result2 = self.linear2(x)
            result3 = self.linear3(x)
        if gnn_style == 'Generalconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x

            result1 = F.log_softmax(x, -1)  # <batch, N, F >
            result2 = self.linear2(x)
            result3 = self.linear3(x)
        if gnn_style == 'Supergatconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x

            result1 = F.log_softmax(x, -1)  # <batch, N, F >
            result2 = self.linear2(x)
            result3 = self.linear3(x)

        return [result1, result2.squeeze(-1), result3.squeeze(-1)]

    def predict(self, data: torch_geometric.data.data.Data, gnn_style):
        x, edge_index = data.x, data.edge_index
        if gnn_style == 'gcn':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            hid = x + original_x
        if gnn_style == 'gat':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            hid = x + original_x

        return hid


class GNN_Origin(nn.Module):
    def __init__(self, data: torch_geometric.data.data.Data, hid_size, gnn_style='gcn', dropout=0.5, alpha=0.2):
        super().__init__()
        self.hid_size = hid_size
        self.dropout = dropout
        self.alpha = alpha
        if gnn_style == 'gcn':
            self.conv1 = GCNConv(data.num_node_features, hid_size)
            self.conv2 = GCNConv(hid_size, hid_size)
        if gnn_style == 'gat':
            self.conv1 = GATConv(data.num_node_features, hid_size, heads=8)
            self.conv2 = GATConv(hid_size * 8, hid_size, heads=1)
        if gnn_style == 'graphsage':
            self.conv1 = SAGEConv(data.num_node_features, hid_size)
            self.conv2 = SAGEConv(hid_size, hid_size)
        if gnn_style == 'transformerconv':
            self.conv1 = TransformerConv(data.num_node_features, hid_size, heads=4)
            self.conv2 = TransformerConv(hid_size * 4, hid_size, heads=1)
        if gnn_style == 'lgconv':
            self.conv1 = LGConv()
            self.lin1 = nn.Linear(data.num_node_features, hid_size)
            self.conv2 = LGConv()
            self.lin2 = nn.Linear(hid_size, data.num_classes)
        if gnn_style == 'ARMAconv':
            self.conv1 = ARMAConv(data.num_node_features, hid_size)
            self.conv2 = ARMAConv(hid_size, hid_size)
        if gnn_style == "Chebconv":
            self.conv1 = ChebConv(data.num_node_features, hid_size, K=3)
            self.conv2 = ChebConv(hid_size, hid_size, K=3)
        if gnn_style == "Generalconv":
            self.conv1 = GeneralConv(data.num_node_features, hid_size)
            self.conv2 = GeneralConv(hid_size, hid_size)
        if gnn_style == "Supergatconv":
            self.conv1 = SuperGATConv(data.num_node_features, hid_size, heads=8)
            self.conv2 = SuperGATConv(hid_size * 8, hid_size, heads=1)
        # node classification linear
        self.linear1 = nn.Linear(hid_size, data.num_classes)
        self.down_sample = nn.Linear(data.num_node_features, hid_size)

    def forward(self, data: torch_geometric.data.data.Data, gnn_style):
        x, edge_index = data.x, data.edge_index
        if gnn_style == 'gcn':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.linear1(x)
            result1 = F.log_softmax(x, -1)  # <batch, N, F >

        if gnn_style == 'gat':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.linear1(x)
            result1 = F.log_softmax(x, -1)  # <batch, N, F >

        if gnn_style == 'graphsage':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.linear1(x)
            result1 = F.log_softmax(x, -1)  # <batch, N, F >

        if gnn_style == 'transformerconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.linear1(x)
            result1 = F.log_softmax(x, 1)
        if gnn_style == 'lgconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.lin2(x)
            result1 = F.log_softmax(x, 1)
        if gnn_style == 'ARMAconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.linear1(x)
            result1 = F.log_softmax(x, -1)  # <batch, N, F >
        if gnn_style == 'Chebconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.linear1(x)
            result1 = F.log_softmax(x, -1)
        if gnn_style == 'Generalconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.linear1(x)
            result1 = F.log_softmax(x, -1)
        if gnn_style == 'Supergatconv':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            x = self.linear1(x)
            result1 = F.log_softmax(x, -1)

        return result1

    def predict(self, data: torch_geometric.data.data.Data, gnn_style):
        x, edge_index = data.x, data.edge_index
        if gnn_style == 'gcn':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            hid = self.linear1(x)
        if gnn_style == 'gat':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            hid = self.linear1(x)
        if gnn_style == 'graphsage':
            original_x = self.down_sample(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x + original_x
            hid = self.linear1(x)

        return hid
