import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        
        #//self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.in_features = in_features   # 节点表示向量的输入特征维度
        self.out_features = out_features   # 节点表示向量的输出特征维度
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) # xavier初始化
        
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414) # xavier初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        #//self.act = nn.PReLU() if act == 'prelu' else act



    '''
    def forward(self, seq, adj):
        Wh = torch.mm(torch.squeeze(seq, 0), self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        
        # 每一个节点和所有节点的attention值        
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        
        e = self.leakyrelu(e)
        zero_vec = -9e15*torch.ones_like(e)
        adj = torch.tensor(adj)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime =  torch.unsqueeze(torch.matmul(attention, Wh), 0)
        
        return  F.elu(h_prime)
    '''

    ''' NO BUG JUST WASTE TOO MUCH MEM AND TIME
    '''
    
    # TODO LET US DEBUG
    def forward(self, h, adj):
        #//hh = torch.squeeze(h, 0)   
        #//Wh = torch.mm(hh, self.W)  
        #@ https://zhuanlan.zhihu.com/p/99927545
        #@ https://blog.csdn.net/weixin_43476533/article/details/107229242
        #@ https://zhuanlan.zhihu.com/p/374914494
        
        #//print(h.shape)
        #//print(adj.shape)
        Wh = torch.mm(torch.squeeze(h, 0), self.W)
        #//print(self.W.shape)
        #//print(Wh.shape)
                
        N = Wh.size()[0]            # number of nodes 2708 
        
        #% Wh.repeat(1, N).view(N*N, -1)  ==>> (N * N, out_features)   
        #% Wh.repeat(N, 1)  ==>> (N * N, out_features)   
        
        a_input = torch.cat([Wh.repeat(1, N).view(N*N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)      
        e = torch.matmul(a_input, self.a).squeeze(2)
        
        e = self.leakyrelu(e)
        zero_vec = -9e15*torch.ones_like(e)
        adj = torch.tensor(adj)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime =  torch.unsqueeze(torch.matmul(attention, Wh), 0)
        
        return F.elu(h_prime)
    





# LET US DEBUG
'''
    def forward(self, h, adj):
        # h: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        # Wh: output_fea [N, out_features]
        #!
        #//Wh = torch.mm(h, self.W) 
        
        #//print("h", type(h))        # h <class 'torch.Tensor'>
        #//print(h.shape)             # torch.Size([1, 2708, 1433]) 
        hh = torch.squeeze(h, 0)   
        #//print("hh", type(hh))      # hh <class 'torch.Tensor'>
        #//print(hh.shape)            # torch.Size([2708, 1433])
        #//print("W", type(self.W))   # W <class 'torch.nn.parameter.Parameter'>
        #//print(self.W.shape)        # torch.Size([1433, 512])
        Wh = torch.mm(hh, self.W)  
        #//print("Wh", type(Wh))      # Wh <class 'torch.Tensor'>
        #//print(Wh.shape)            # torch.Size([2708, 512])
        
        # 实现论文中的特征拼接操作 Wh_i || Wh_j  得到 shape = (N, N, 2 * out_features) 新特征矩阵
        
        N = Wh.size()[0]            # number of nodes 
        #//print("OK1")  
        print("N", N)
        #% Wh.repeat(1, N).view(N*N, -1)  ==>> (N * N, out_features)   
        #% Wh.repeat(N, 1)  ==>> (N * N, out_features)   
        a_input = torch.cat([Wh.repeat(1, N).view(N*N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        #//print("OK2")          
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        #//print("OK3") 
        
        zero_vec = -9e15*torch.ones_like(e)
        #//print("OK4") 
        #//print(type(zero_vec))# <class 'torch.Tensor'>
        #//print(type(e))       # <class 'torch.Tensor'>
        #//print(type(adj))     # <class 'numpy.matrix'>
        adj = torch.tensor(adj)
        #//print("OK5") 
        #//print(type(adj))
        #!
        attention = torch.where(adj > 0, e, zero_vec)
        #//print("OK6") 
        attention = F.softmax(attention, dim=1)
        #//print("OK7") 
        attention = F.dropout(attention, self.dropout, training=self.training)
        #//print("OK8") 
        
        #!h_prime = torch.matmul(attention, Wh)
        h_prime =  torch.unsqueeze(torch.matmul(attention, Wh), 0)
        
        #//print("OK9") 
        print("h_prime", type(h_prime))
        #//print("OK10") 
        print(h_prime.shape)
        
        if self.concat:
            #//print("concat OKK")
            return F.elu(h_prime)
        else:
            return h_prime
'''

















# Origin GCN layer Frame
'''
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)
'''