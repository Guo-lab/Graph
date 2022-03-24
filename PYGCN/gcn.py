# https://arxiv.org/pdf/1609.02907.pdf
# 
# https://github.com/tkipf/pygcn
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution



class GCN(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        

    # 整个网络的前向传播的方式：
    # relu(gc1) --> dropout --> gc2 --> log_softmax
    def forward(self, x, adj):
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        return F.log_softmax(x, dim=1)