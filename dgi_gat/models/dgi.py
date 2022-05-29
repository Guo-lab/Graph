import torch
import torch.nn as nn
from layers import GraphAttention, AvgReadout, Discriminator # , GraphAttention2



class DGI(nn.Module):
    def __init__(self, n_in, n_h):
        super(DGI, self).__init__()
        
        self.gat = GraphAttention(n_in, n_h, 0.5, 0.2)               # self, in_features, out_features, dropout, alpha, concat=True
        #//print("gat init OK")
        #//self.gat2 = GraphAttention2(n_in, n_h, 0.5, 0.2)         
        
        self.read = AvgReadout()
        #//print("AvgReadout init OK")
        self.sigm = nn.Sigmoid()
        #//print("Sigmoid init OK")
        self.disc = Discriminator(n_h)
        #//print("Discriminator init OK")

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2):
        h_1 = self.gat(seq1, adj)
        #//print("h1 gat OK") 
        # SHOULD BE                       # torch.Size([1, 2708, 512])
        #//print(h_1.shape)                  # torch.Size([2708, 512])
        #//print("seq shape, adj shape", seq1.shape, adj.shape)
        
        c = self.read(h_1, msk)
        #//print("AvgReadout OK")   
        # SHOULD BE                       # torch.Size([1, 512])
        #//print(c.shape)                    # torch.Size([2708])
        c = self.sigm(c)
        #//print("Sigmoid OK")
        
        #//print("seq2 shape, adj shape", seq2.shape, adj.shape)
        h_2 = self.gat(seq2, adj)
        #//print(h_2.shape)
        #//print("h2 gat OK")
        
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)        
        #ret = self.disc(c, h_1, h_1, samp_bias1, samp_bias2)
        #//print("Discriminator OK")
        return ret

    # Detach the return variables
    def embed(self, seq, adj, msk):
        h_1 = self.gat(seq, adj)
        #//print("h1 embed OK")
        
        c = self.read(h_1, msk)
        #//print("h1 AvgReadout OK")
        return h_1.detach(), c.detach()