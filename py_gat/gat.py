import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphAttentionLayer



class GAT(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """ Dense version of GAT.
            n_heads 表示有几个GAL层,最后进行拼接在一起,类似self-attention
            从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        
        self.dropout = dropout
        
        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            
            #% add_module 是 Module类的成员函数
            # 输入参数为 Module.add_module(name: str, module: Module)
            # 功能为 Module添加一个子module，对应名字为name
            # add_module() 函数也可以在GAT.init(self)以外定义A的子模块
            self.add_module('attention_{}'.format(i), attention)

        #% 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        # 输出层的输入张量的 shape 为(nhid * nheads, nclass)
        # 是因为在forward函数中多个注意力机制在同一个节点上得到的多个不同特征被拼接成了一个长的特征
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)


    # multi-head
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 将多个注意力机制在同一个节点上得到的多个不同特征进行拼接形成一个长特征
        # 即将每个head得到的表示进行拼接
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        
        # F.log_softmax在数学上等价于log(softmax(x))，
        # 但做这两个单独操作速度较慢，数值上也不稳定。
        # log_softmax速度变快，保持数值稳定，正确计算输出和梯度
        return F.log_softmax(x, dim=1)