import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphAttentionLayer(nn.Module):
    # Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        
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


    def forward(self, h, adj):
        # h: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        # Wh: output_fea [N, out_features]
        # adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        Wh = torch.mm(h, self.W) 
    
        # 实现论文中的特征拼接操作 Wh_i || Wh_j 
        # 公式解读： https://zhuanlan.zhihu.com/p/81350196
        # 得到 shape = (N, N, 2 * out_features) 新特征矩阵
        
        # number of nodes
        N = Wh.size()[0]    
        
        #repeat方法可以对 Wh 张量中的单维度和非单维度进行复制操作，并且会真正的复制数据保存到内存中
        #repeat(N, 1)表示dim=0维度的数据复制N份，dim=1维度的数据保持不变

        #% Wh.repeat(1, N) 
        # (N*F)FF...F
        #   '------' => N times
        #% Wh.repeat(1, N).view(N*N, -1)  ==>> (N * N, out_features)   
        # https://www.jianshu.com/p/a2102492293a   
        #% Wh.repeat(N, 1)  ==>> (N * N, out_features)   
        # N * F, -
        #  ... ,  | => N times 
        # N * F, -  
        
        # e1 || e1
        # e1 || e2
        # ...
        # e1 || eN
        # e2 || e1
        # ...
        # eN || e3
        # ...
        # eN || eN
        
        # [N, N, 2*out_features]
        a_input = torch.cat([Wh.repeat(1, N).view(N*N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        
        # [N, N, 1] => squeeze(2) => [N, N] 图注意力的相关系数（未归一化）        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
 

        # 将没有连接的边置为负无穷
        zero_vec = -9e15*torch.ones_like(e)
        
        #% attention [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = torch.where(adj > 0, e, zero_vec)
        # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.softmax(attention, dim=1)
        # dropout，防止过拟合
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime



    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
    
    
#* Origin
''' 
    def forward(self, h, adj):

        Wh = torch.mm(h, self.W) 
        
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        
        # broadcast add
        e = Wh1 + Wh2.t()
        
        e = self.leakyrelu(e)

        zero_vec = -9e15*torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        ...
        .
'''
#* Or 
# torch.repeat_interleave()