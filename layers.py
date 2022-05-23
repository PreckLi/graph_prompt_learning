import torch
from torch import nn

class Degree_layer(nn.Module):
    def __init__(self,inp_size,out_size):
        super(Degree_layer, self).__init__()
        self.linear=nn.Linear(inp_size,out_size)

    def forward(self,x):
        x=torch.sigmoid(x)
        x=self.linear(x)
        return x