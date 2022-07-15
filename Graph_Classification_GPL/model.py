import torch
import torch_geometric
from torch_geometric.nn import GCNConv, GINConv, TopKPooling, SAGPooling, EdgePooling
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_sort_pool, SAGEConv, ASAPooling, DenseSAGEConv, \
    dense_diff_pool
from torch_geometric.nn import GraphConv, JumpingKnowledge, BatchNorm as BN
from math import ceil


class GNN(nn.Module):
    def __init__(self, data: torch_geometric.data.data.Data, hid_size, gnn_style='sagpool', dropout=0.5,
                 pooling_ratio=0.2):
        super().__init__()
        self.num_feats = data.num_node_features
        self.nhid = hid_size
        self.num_classes = data.num_classes
        self.pool_ratio = pooling_ratio
        self.dropout = dropout
        if gnn_style == 'sagpool':
            self.conv1 = GCNConv(self.num_feats, self.nhid)
            self.pool1 = SAGPooling(self.nhid, ratio=self.pool_ratio)
            self.conv2 = GCNConv(self.nhid, self.nhid)
            self.pool2 = SAGPooling(self.nhid, ratio=self.pool_ratio)
            self.conv3 = GCNConv(self.nhid, self.nhid)
            self.pool3 = SAGPooling(self.nhid, ratio=self.pool_ratio)

            self.lin1 = nn.Linear(self.nhid * 2, self.nhid)
            self.lin2 = nn.Linear(self.nhid, self.nhid // 2)
            self.lin3 = nn.Linear(self.nhid // 2, self.num_classes)
            # node's degree distribution linear
            self.lin5 = nn.Linear(self.nhid // 2, data.graph_degree_distribution.shape[1])
        if gnn_style == 'topkpool':
            self.pool_ratio = 0.8
            self.conv1 = GraphConv(self.num_feats, self.nhid, aggr='mean')
            self.convs = torch.nn.ModuleList()
            self.pools = torch.nn.ModuleList()
            self.num_layers = 2
            self.convs.extend([
                GraphConv(self.nhid, self.nhid, aggr='mean')
                for i in range(self.num_layers - 1)
            ])
            self.pools.extend(
                [TopKPooling(self.nhid, self.pool_ratio) for i in range((self.num_layers) // 2)])
            self.jump = JumpingKnowledge(mode='cat')
            self.lin1 = nn.Linear(self.num_layers * self.nhid, self.nhid)
            self.lin2 = nn.Linear(self.nhid, self.num_classes)
            # node neighborhoods' degree prediction linear
            self.lin4 = nn.Linear(self.nhid, data.graph_degree_distribution.shape[1])
        if gnn_style == 'edgepool':
            self.conv1 = GraphConv(self.num_feats, self.nhid, aggr='mean')
            self.convs = torch.nn.ModuleList()
            self.pools = torch.nn.ModuleList()
            self.num_layers = 2
            self.convs.extend([
                GraphConv(self.nhid, self.nhid, aggr='mean')
                for i in range(self.num_layers - 1)
            ])
            self.pools.extend(
                [EdgePooling(self.nhid) for i in range((self.num_layers) // 2)])
            self.jump = JumpingKnowledge(mode='cat')
            self.lin1 = nn.Linear(self.num_layers * self.nhid, self.nhid)
            self.lin2 = nn.Linear(self.nhid, self.num_classes)
            # node neighborhoods' degree prediction linear
            self.lin4 = nn.Linear(self.nhid, data.graph_degree_distribution.shape[1])
        if gnn_style[:3] == 'gin':
            if gnn_style != 'gin' and gnn_style[3] == '0':
                train_eps = False
            if gnn_style[:3] == 'gin':
                train_eps = True
            self.num_layers = 2
            self.conv1 = GINConv(
                nn.Sequential(
                    nn.Linear(self.num_feats, self.nhid),
                    nn.ReLU(),
                    nn.Linear(self.nhid, self.nhid),
                    nn.ReLU(),
                    BN(self.nhid),
                ), train_eps=train_eps)
            self.convs = torch.nn.ModuleList()
            for i in range(self.num_layers - 1):
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(self.nhid, self.nhid),
                            nn.ReLU(),
                            nn.Linear(self.nhid, self.nhid),
                            nn.ReLU(),
                            BN(self.nhid),
                        ), train_eps=train_eps))
            if gnn_style[-2:] != 'jk':
                # without jumping knowledge
                self.lin1 = nn.Linear(self.nhid, self.nhid)
            if gnn_style[-2:] == 'jk':
                # with jumping knowledge
                self.jump = JumpingKnowledge('cat')
                self.lin1 = nn.Linear(self.num_layers * self.nhid, self.nhid)

            self.lin2 = nn.Linear(self.nhid, self.num_classes)
            # node neighborhoods' degree prediction linear
            self.lin4 = nn.Linear(self.nhid, data.graph_degree_distribution.shape[1])
        if gnn_style == 'sortpool':
            self.num_layers = 3
            self.k = 30
            self.conv1 = SAGEConv(self.num_feats, self.nhid)
            self.convs = torch.nn.ModuleList()
            for i in range(self.num_layers - 1):
                self.convs.append(SAGEConv(self.nhid, self.nhid))
            self.conv1d = nn.Conv1d(self.nhid, 32, 5)
            self.lin1 = nn.Linear(32 * (self.k - 5 + 1), self.nhid)
            self.lin2 = nn.Linear(self.nhid, self.num_classes)
            # node neighborhoods' degree prediction linear
            self.lin4 = nn.Linear(self.nhid, data.graph_degree_distribution.shape[1])
        if gnn_style == 'asapool':
            num_layers = 3
            ratio = 0.8
            self.conv1 = GraphConv(self.num_feats, hid_size, aggr='mean')
            self.convs = torch.nn.ModuleList()
            self.pools = torch.nn.ModuleList()
            self.convs.extend([
                GraphConv(hid_size, hid_size, aggr='mean')
                for i in range(num_layers - 1)
            ])
            self.pools.extend([
                ASAPooling(hid_size, ratio, dropout=dropout)
                for i in range((num_layers) // 2)
            ])
            self.jump = JumpingKnowledge(mode='cat')
            self.lin1 = nn.Linear(num_layers * hid_size, hid_size)
            self.lin2 = nn.Linear(hid_size, self.num_classes)
            # node neighborhoods' degree prediction linear
            self.lin4 = nn.Linear(self.nhid, data.graph_degree_distribution.shape[1])
        self.loss_linear = nn.Linear(self.nhid, 1)

    def forward(self, data: torch_geometric.data.Data, gnn_style):
        x, edge_index, batch, degree = data.x, data.edge_index, data.batch, data.degree
        if gnn_style == 'sagpool':
            x = F.relu(self.conv1(x, edge_index))
            mse_x = self.loss_linear(x)
            mse_loss = F.mse_loss(mse_x, degree.view(-1, 1))
            x, edge_index, _, batch, _, _ = self.pool1.forward(x, edge_index, None, batch)
            # readout
            x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2.forward(x, edge_index, None, batch)
            # readout
            x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3.forward(x, edge_index, None, batch)
            # readout
            x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = x1 + x2 + x3

            x = F.relu(self.lin1(x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.lin2(x))
            result1 = F.log_softmax(self.lin3(x), dim=1)
            result3 = self.lin5(x)

        if gnn_style == 'topkpool':
            x = F.relu(self.conv1(x, edge_index))
            xs = [global_mean_pool(x, batch)]
            for i, conv in enumerate(self.convs):
                x = F.relu(conv(x, edge_index))
                xs += [global_mean_pool(x, batch)]
                if i % 2 == 0 and i < len(self.convs) - 1:
                    pool = self.pools[i // 2]
                    x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)
            mse_x = self.loss_linear(x)
            mse_loss = F.mse_loss(mse_x, degree.view(-1, 1))
            x = self.jump(xs)
            x = F.relu(self.lin1(x))
            classify_x = F.dropout(x, p=0.5, training=self.training)
            classify_x = self.lin2(classify_x)
            result1 = F.log_softmax(classify_x, dim=-1)
            result3 = self.lin4(x).squeeze(-1)

        if gnn_style == 'edgepool':
            x = F.relu(self.conv1(x, edge_index))
            xs = [global_mean_pool(x, batch)]
            for i, conv in enumerate(self.convs):
                x = F.relu(conv(x, edge_index))
                xs += [global_mean_pool(x, batch)]
                if i % 2 == 0 and i < len(self.convs) - 1:
                    pool = self.pools[i // 2]
                    x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
            mse_x = self.loss_linear(x)
            mse_loss = F.mse_loss(mse_x, degree.view(-1, 1))
            x = self.jump(xs)
            x = F.relu(self.lin1(x))
            classify_x = F.dropout(x, p=0.5, training=self.training)
            classify_x = self.lin2(classify_x)
            result1 = F.log_softmax(classify_x, dim=-1)
            result3 = self.lin4(x).squeeze(-1)
        if gnn_style[:3] == 'gin':
            x = self.conv1(x, edge_index)
            if gnn_style[-2:] != 'jk':
                # pure gin
                for conv in self.convs:
                    x = conv(x, edge_index)
            mse_x = self.loss_linear(x)
            mse_loss = F.mse_loss(mse_x, degree.view(-1, 1))
            if gnn_style[-2:] == 'jk':
                # gin with JK
                xs = [x]
                for conv in self.convs:
                    x = conv(x, edge_index)
                    xs += [x]
                x = self.jump(xs)
            x = global_mean_pool(x, batch)
            x = F.relu(self.lin1(x))
            classify_x = F.dropout(x, p=0.5, training=self.training)
            classify_x = self.lin2(classify_x)
            result1 = F.log_softmax(classify_x, dim=-1)
            result3 = self.lin4(x).squeeze(-1)
        if gnn_style == 'sortpool':
            x = F.relu(self.conv1(x, edge_index))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            mse_x = self.loss_linear(x)
            mse_loss = F.mse_loss(mse_x, degree.view(-1, 1))
            x = global_sort_pool(x, batch, self.k)
            x = x.view(len(x), self.k, -1).permute(0, 2, 1)
            x = F.relu(self.conv1d(x))
            x = x.view(len(x), -1)
            x = F.relu(self.lin1(x))
            classify_x = F.dropout(x, p=0.5, training=self.training)
            result1 = self.lin2(classify_x)
            result1 = F.log_softmax(result1, dim=-1)
            result3 = self.lin4(x).squeeze(-1)
        if gnn_style == 'asapool':
            edge_weight = None
            x = F.relu(self.conv1(x, edge_index))
            mse_x = self.loss_linear(x)
            mse_loss = F.mse_loss(mse_x, degree.view(-1, 1))
            xs = [global_mean_pool(x, batch)]
            for i, conv in enumerate(self.convs):
                x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
                x = F.relu(x)
                xs += [global_mean_pool(x, batch)]
                if i % 2 == 0 and i < len(self.convs) - 1:
                    pool = self.pools[i // 2]
                    x, edge_index, edge_weight, batch, _ = pool(
                        x=x, edge_index=edge_index, edge_weight=edge_weight,
                        batch=batch)
            x = self.jump(xs)
            x = F.relu(self.lin1(x))
            classify_x = F.dropout(x, p=0.5, training=self.training)
            result1 = self.lin2(classify_x)
            result1 = F.log_softmax(result1, dim=-1)
            result3 = self.lin4(x).squeeze(-1)

        return [result1, mse_loss, result3]

    def predict(self, data: torch_geometric.data.Data, gnn_style):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if gnn_style == 'sagpool':
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1.forward(x, edge_index, None, batch)
            # readout
            x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2.forward(x, edge_index, None, batch)
            # readout
            x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3.forward(x, edge_index, None, batch)
            # readout
            x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = x1 + x2 + x3

            x = F.relu(self.lin1(x))
            x = F.dropout(x, self.dropout, training=self.training)
            hid = self.lin2(x)
        return hid


class GNN_Origin(nn.Module):
    def __init__(self, data: torch_geometric.data.data.Data, hid_size, gnn_style='sagpool', dropout=0.5,
                 pooling_ratio=0.2):
        super().__init__()
        self.num_feats = data.num_node_features
        self.nhid = hid_size
        self.num_classes = data.num_classes
        self.pool_ratio = pooling_ratio
        self.dropout = dropout
        if gnn_style == 'sagpool':
            self.conv1 = GCNConv(self.num_feats, self.nhid)
            self.pool1 = SAGPooling(self.nhid, ratio=self.pool_ratio)
            self.conv2 = GCNConv(self.nhid, self.nhid)
            self.pool2 = SAGPooling(self.nhid, ratio=self.pool_ratio)
            self.conv3 = GCNConv(self.nhid, self.nhid)
            self.pool3 = SAGPooling(self.nhid, ratio=self.pool_ratio)

            self.lin1 = nn.Linear(self.nhid * 2, self.nhid)
            self.lin2 = nn.Linear(self.nhid, self.nhid // 2)
            self.lin3 = nn.Linear(self.nhid // 2, self.num_classes)
        if gnn_style == 'topkpool':
            self.pool_ratio = 0.8
            self.conv1 = GraphConv(self.num_feats, self.nhid, aggr='mean')
            self.convs = torch.nn.ModuleList()
            self.pools = torch.nn.ModuleList()
            self.num_layers = 2
            self.convs.extend([
                GraphConv(self.nhid, self.nhid, aggr='mean')
                for i in range(self.num_layers - 1)
            ])
            self.pools.extend(
                [TopKPooling(self.nhid, self.pool_ratio) for i in range((self.num_layers) // 2)])
            self.jump = JumpingKnowledge(mode='cat')
            self.lin1 = nn.Linear(self.num_layers * self.nhid, self.nhid)
            self.lin2 = nn.Linear(self.nhid, self.num_classes)
        if gnn_style == 'edgepool':
            self.conv1 = GraphConv(self.num_feats, self.nhid, aggr='mean')
            self.convs = torch.nn.ModuleList()
            self.pools = torch.nn.ModuleList()
            self.num_layers = 2
            self.convs.extend([
                GraphConv(self.nhid, self.nhid, aggr='mean')
                for i in range(self.num_layers - 1)
            ])
            self.pools.extend(
                [EdgePooling(self.nhid) for i in range((self.num_layers) // 2)])
            self.jump = JumpingKnowledge(mode='cat')
            self.lin1 = nn.Linear(self.num_layers * self.nhid, self.nhid)
            self.lin2 = nn.Linear(self.nhid, self.num_classes)
        if gnn_style[:3] == 'gin':
            if gnn_style != 'gin' and gnn_style[3] == '0':
                train_eps = False
            if gnn_style[:3] == 'gin':
                train_eps = True
            self.num_layers = 2
            self.conv1 = GINConv(
                nn.Sequential(
                    nn.Linear(self.num_feats, self.nhid),
                    nn.ReLU(),
                    nn.Linear(self.nhid, self.nhid),
                    nn.ReLU(),
                    BN(self.nhid),
                ), train_eps=train_eps)
            self.convs = torch.nn.ModuleList()
            for i in range(self.num_layers - 1):
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(self.nhid, self.nhid),
                            nn.ReLU(),
                            nn.Linear(self.nhid, self.nhid),
                            nn.ReLU(),
                            BN(self.nhid),
                        ), train_eps=train_eps))
            if gnn_style[-2:] != 'jk':
                # without jumping knowledge
                self.lin1 = nn.Linear(self.nhid, self.nhid)
            if gnn_style[-2:] == 'jk':
                # with jumping knowledge
                self.jump = JumpingKnowledge('cat')
                self.lin1 = nn.Linear(self.num_layers * self.nhid, self.nhid)

            self.lin2 = nn.Linear(self.nhid, self.num_classes)
        if gnn_style == 'sortpool':
            self.num_layers = 3
            self.k = 30
            self.conv1 = SAGEConv(self.num_feats, self.nhid)
            self.convs = torch.nn.ModuleList()
            for i in range(self.num_layers - 1):
                self.convs.append(SAGEConv(self.nhid, self.nhid))
            self.conv1d = nn.Conv1d(self.nhid, 32, 5)
            self.lin1 = nn.Linear(32 * (self.k - 5 + 1), self.nhid)
            self.lin2 = nn.Linear(self.nhid, self.num_classes)
            self.reset_parameters_sortpool()
        if gnn_style == 'asapool':
            num_layers = 3
            ratio = 0.8
            self.conv1 = GraphConv(self.num_feats, hid_size, aggr='mean')
            self.convs = torch.nn.ModuleList()
            self.pools = torch.nn.ModuleList()
            self.convs.extend([
                GraphConv(hid_size, hid_size, aggr='mean')
                for i in range(num_layers - 1)
            ])
            self.pools.extend([
                ASAPooling(hid_size, ratio, dropout=dropout)
                for i in range((num_layers) // 2)
            ])
            self.jump = JumpingKnowledge(mode='cat')
            self.lin1 = nn.Linear(num_layers * hid_size, hid_size)
            self.lin2 = nn.Linear(hid_size, self.num_classes)
        if gnn_style == 'diffpool':
            ratio = 0.25
            num_layers = 3
            num_nodes = ceil(ratio * data.num_nodes)
            self.embed_block1 = Block(self.num_feats, hid_size, hid_size)
            self.pool_block1 = Block(self.num_feats, hid_size, num_nodes)

            self.embed_blocks = torch.nn.ModuleList()
            self.pool_blocks = torch.nn.ModuleList()
            for i in range((num_layers // 2) - 1):
                num_nodes = ceil(ratio * num_nodes)
                self.embed_blocks.append(Block(hid_size, hid_size, hid_size))
                self.pool_blocks.append(Block(hid_size, hid_size, num_nodes))

            self.jump = JumpingKnowledge(mode='cat')
            self.lin1 = nn.Linear((len(self.embed_blocks) + 1) * hid_size, hid_size)
            self.lin2 = nn.Linear(hid_size, self.num_classes)

    def forward(self, data: torch_geometric.data.Data, gnn_style):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if gnn_style == 'sagpool':
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1.forward(x, edge_index, None, batch)
            # readout
            x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2.forward(x, edge_index, None, batch)
            # readout
            x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3.forward(x, edge_index, None, batch)
            # readout
            x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = x1 + x2 + x3

            x = F.relu(self.lin1(x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.lin2(x))
            result1 = F.log_softmax(self.lin3(x), dim=1)

        if gnn_style == 'topkpool':
            x = F.relu(self.conv1(x, edge_index))
            xs = [global_mean_pool(x, batch)]
            for i, conv in enumerate(self.convs):
                x = F.relu(conv(x, edge_index))
                xs += [global_mean_pool(x, batch)]
                if i % 2 == 0 and i < len(self.convs) - 1:
                    pool = self.pools[i // 2]
                    x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)
            x = self.jump(xs)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            result1 = F.log_softmax(x, dim=-1)
        if gnn_style == 'edgepool':
            x = F.relu(self.conv1(x, edge_index))
            xs = [global_mean_pool(x, batch)]
            for i, conv in enumerate(self.convs):
                x = F.relu(conv(x, edge_index))
                xs += [global_mean_pool(x, batch)]
                if i % 2 == 0 and i < len(self.convs) - 1:
                    pool = self.pools[i // 2]
                    x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
            x = self.jump(xs)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            result1 = F.log_softmax(x, dim=-1)
        if gnn_style[:3] == 'gin':
            x = self.conv1(x, edge_index)
            if gnn_style[-2:] != 'jk':
                # pure gin
                for conv in self.convs:
                    x = conv(x, edge_index)
            if gnn_style[-2:] == 'jk':
                # gin with JK
                xs = [x]
                for conv in self.convs:
                    x = conv(x, edge_index)
                    xs += [x]
                x = self.jump(xs)
            x = global_mean_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            result1 = F.log_softmax(x, dim=-1)
        if gnn_style == 'sortpool':
            x = F.relu(self.conv1(x, edge_index))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            x = global_sort_pool(x, batch, self.k)
            x = x.view(len(x), self.k, -1).permute(0, 2, 1)
            x = F.relu(self.conv1d(x))
            x = x.view(len(x), -1)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            result1 = F.log_softmax(self.lin2(x), dim=-1)
        if gnn_style == 'asapool':
            edge_weight = None
            x = F.relu(self.conv1(x, edge_index))
            xs = [global_mean_pool(x, batch)]
            for i, conv in enumerate(self.convs):
                x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
                x = F.relu(x)
                xs += [global_mean_pool(x, batch)]
                if i % 2 == 0 and i < len(self.convs) - 1:
                    pool = self.pools[i // 2]
                    x, edge_index, edge_weight, batch, _ = pool(
                        x=x, edge_index=edge_index, edge_weight=edge_weight,
                        batch=batch)
            x = self.jump(xs)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            result1 = F.log_softmax(self.lin2(x), dim=-1)
        if gnn_style == 'diffpool':
            s = self.pool_block1(x, edge_index, batch)
            x = F.relu(self.embed_block1(x, edge_index, batch))
            xs = [x.mean(dim=1)]
            x, edge_index, _, _ = dense_diff_pool(x, edge_index, s, batch)

            for i, (embed_block, pool_block) in enumerate(
                    zip(self.embed_blocks, self.pool_blocks)):
                s = pool_block(x, edge_index)
                x = F.relu(embed_block(x, edge_index))
                xs.append(x.mean(dim=1))
                if i < len(self.embed_blocks) - 1:
                    x, edge_index, _, _ = dense_diff_pool(x, edge_index, s)

            x = self.jump(xs)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            result1 = self.lin2(x)

        return result1

    def predict(self, data: torch_geometric.data.Data, gnn_style):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if gnn_style == 'sagpool':
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1.forward(x, edge_index, None, batch)
            # readout
            x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2.forward(x, edge_index, None, batch)
            # readout
            x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3.forward(x, edge_index, None, batch)
            # readout
            x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = x1 + x2 + x3

            x = F.relu(self.lin1(x))
            x = F.dropout(x, self.dropout, training=self.training)
            hid = self.lin2(x)
        return hid

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.jump:
            self.jump.reset_parameters()

    def reset_parameters_sortpool(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mode='cat'):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin = nn.Linear(hidden_channels + out_channels, out_channels)
        else:
            self.lin = nn.Linear(out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None):
        x1 = F.relu(self.conv1(x, adj, mask))
        x2 = F.relu(self.conv2(x1, adj, mask))
        return self.lin(self.jump([x1, x2]))
