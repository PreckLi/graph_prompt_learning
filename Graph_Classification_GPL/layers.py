from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
import torch
from torch import nn

class SAGPool(nn.Module):
    def __init__(self,inp_size,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.inp_size=inp_size
        self.ratio=ratio
        self.score_layer=Conv(inp_size,1)
        self.non_linear=non_linearity

    def forward(self,x,edge_index,edge_x=None,batch=None):
        if batch is None:
            batch=edge_index.new_zeros(x.size(0))

        score=self.score_layer(x,edge_index).squeeze()
        perm=topk(score,self.ratio,batch)
        x=x[perm]*self.non_linear(score[perm].view(-1,1))
        batch=batch[perm]
        edge_index,edge_x=filter_adj(edge_index,edge_x,perm,num_nodes=score.size(0))

        return x,edge_index,edge_x,batch,perm

