import torch
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch import nn
from torch.nn import functional as F
from layers import Degree_layer


class GNN(nn.Module):
    def __init__(self, data: torch_geometric.data.data.Data, hid_size, gnn_style='gcn', dropout=0.5, alpha=0.2):
        super(GNN, self).__init__()
        self.hid_size = hid_size
        self.dropout = dropout
        self.alpha = alpha
        if gnn_style == 'gcn':
            self.conv1 = GCNConv(data.num_node_features, hid_size)
            self.conv2 = GCNConv(hid_size, hid_size)
        if gnn_style == 'gat':
            self.conv1 = GATConv(data.num_node_features, hid_size, heads=8)
            self.conv2 = GATConv(hid_size * 8, hid_size, heads=1)
        if gnn_style == 'gin':
            self.conv1 = GINConv([data.num_node_features, hid_size])
            self.conv2 = GINConv([hid_size, hid_size])
        if gnn_style == 'graphsage':
            self.conv1 = SAGEConv(data.num_node_features, hid_size)
            self.conv2 = SAGEConv(hid_size, hid_size)
        # node classification linear
        self.linear1 = nn.Linear(hid_size, data.num_classes)
        # node degree prediction linear
        self.linear2 = Degree_layer(hid_size, 1)
        # node neighborhoods' degree prediction linear
        self.linear3 = Degree_layer(hid_size, 1)

        self.down_sample = nn.Linear(data.num_node_features, hid_size)

    def forward(self, data: torch_geometric.data.data.Data):
        x, edge_index = data.x, data.edge_index
        original_x = self.down_sample(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = x + original_x

        result1 = F.log_softmax(self.linear1(x), 1)
        result2 = self.linear2(x)
        result3 = self.linear3(x)

        return [result1, result2.squeeze(-1), result3.squeeze(-1)]
