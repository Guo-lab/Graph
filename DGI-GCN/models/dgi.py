import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        #//print("h1 gat OK") 
        #//print("h1 shape", h_1.shape)   
        #//print("seq shape, adj shape", seq1.shape, adj.shape)     
        
        c = self.read(h_1, msk)
        #//print("AvgReadout OK")   
        #//print("c shape", c.shape)             
        c = self.sigm(c)

        #//print("seq2 shape, adj shape", seq2.shape, adj.shape)
        h_2 = self.gcn(seq2, adj, sparse)
        #//print(h_2.shape)
        #//print("h2 gat OK")
        
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()