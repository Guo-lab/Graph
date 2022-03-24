import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class GraphConvolution(Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        #* call
        self.reset_parameters()


    def reset_parameters(self):
        # self.weight.size(1)是weightShape(in_features, out_features)的out_features
        stdv = 1. / math.sqrt(self.weight.size(1))
        
        # 均匀分布
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    # %特征矩阵feature为H 邻接矩阵adj为A 权重为W 则输出
    # step1. 求 H 和 W 乘积  HW
    # step2. 求 A 和 HW 乘积 AHW，
    #      这里 A = D_A^-1 · (A+I) 并归一化，
    #          H = D_H^-1 · H 
    # %dimension
    #   adj 	2708,2708  A
    # features 	2708,1433  H0
    #  labels	2708, 0~6
    # 第一次gc后 2708,nhid
    # 第二次gc后 2708,7 (7个类别)
    def forward(self, input, adj):
        # 2D 矩阵乘法 - input * self.weight (3D[with batch] torch.matmul)
        support = torch.mm(input, self.weight)
        # 矩阵乘法 sparse * dense OR dense * dense
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


    # return Class Introduction
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'