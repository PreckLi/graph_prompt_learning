import torch
from torch import nn


class Degree_layer(nn.Module):
    def __init__(self, inp_size, out_size):
        super(Degree_layer, self).__init__()
        self.linear = nn.Linear(inp_size, out_size)

    def forward(self, x):
        x = torch.sigmoid(x)
        x = self.linear(x)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_size = in_size
        self.out_size = out_size
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_size, out_size)))
        nn.init.xavier_uniform_(self.W.data, 1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_size, 1)))
        nn.init.xavier_uniform_(self.W.data, 1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_size, :])  # (num_node,1)
        Wh2 = torch.matmul(Wh, self.a[self.out_size:, :])  # (num_node,1)

        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # (num_node,out_size)
        e = self._prepare_attentional_mechanism_input(Wh)  # (num_node,num_node)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
