import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,Degree_layer

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_prompt(nn.Module):
    def __init__(self, nfeat, nhid, nclass, degree_max,dropout):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3=GraphConvolution(nhid,nhid)
        self.dropout = dropout

        # node classification linear
        self.linear1 = nn.Linear(nhid, nclass)
        # node degree prediction linear
        self.linear2 = Degree_layer(nhid, 1)
        # node neighborhoods' degree prediction linear
        self.linear3 = nn.Linear(nhid,degree_max)

        self.down_sample = nn.Linear(nfeat, nhid)

    def forward(self, x, adj):
        original_x = self.down_sample(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        classify_x = self.gc2(x, adj)
        x = F.relu(self.gc3(x,adj))
        x = x + original_x

        result1=F.log_softmax(classify_x, dim=1)
        result2 = self.linear2(x).squeeze(-1)
        result3 = self.linear3(x)

        return [result1, result2, result3]