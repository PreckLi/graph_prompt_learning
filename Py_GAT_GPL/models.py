import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, DegreeLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GAT_prompt(nn.Module):
    def __init__(self, nfeat, nhid, nclass,degree_max, dropout, alpha, nheads):
        """Dense version of GAT."""
        super().__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att1 = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_att2 = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

        # node classification linear
        self.linear1 = nn.Linear(nhid, nclass)
        # node degree prediction linear
        self.linear2 = DegreeLayer(nhid, 1)
        # node neighborhoods' degree prediction linear
        self.linear3 = nn.Linear(nhid,degree_max)

        self.down_sample = nn.Linear(nfeat, nhid)

    def forward(self, x, adj):
        original_x = self.down_sample(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        classify_x = F.elu(self.out_att1(x, adj))
        x = self.out_att2(x,adj)
        x = x + original_x

        result1 = F.log_softmax(classify_x, dim=1)
        result2 = self.linear2(x).squeeze(-1)
        result3 = self.linear3(x)

        return [result1, result2, result3]