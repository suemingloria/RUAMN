#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layer_norm import *
import torch.nn.init as init
import numpy as np

class RUAMN(nn.Module):
    def __init__(self, feature_size=1024, embed_size=512, max_video_len=1600, hops=4, dropout=0.2, te=True, n_segments=6):
        super(RUAMN, self).__init__()
        self.hops = hops
        self.temporal_encoding = te
        self.n_segments = n_segments
        self.embed_size = embed_size

        self.gru = nn.GRU(feature_size, feature_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal_(param)

        self.dropout = nn.Dropout(p=dropout)
        self.A1 = nn.ModuleList([nn.Linear(feature_size, embed_size,bias=False) for _ in range((hops+1)*(n_segments//2))])
        self.A2 = nn.ModuleList([nn.Linear(feature_size, embed_size,bias=False) for _ in range((hops+1)*(n_segments//2))])
        self.memory_update1 = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(hops*(n_segments//2))])
        self.memory_update2 = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(hops*(n_segments//2))])

        self.layer_norm1 = nn.ModuleList([LayerNorm(embed_size) for _ in range(hops*(n_segments//2))])
        self.layer_norm2 = nn.ModuleList([LayerNorm(embed_size) for _ in range(hops*(n_segments//2))])

        self.linear_x = nn.Linear(feature_size,embed_size)
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.ka = nn.Linear(embed_size, embed_size)
        self.layer_norm_y = LayerNorm(embed_size)
        self.layer_norm_ka = LayerNorm(embed_size)

        self.linear = nn.Linear(embed_size, 1)
        self.sig = nn.Sigmoid()

        if self.temporal_encoding:
            self.TA = nn.Parameter(torch.Tensor(max_video_len, embed_size).normal_(0, 0.1))
            self.TC = nn.Parameter(torch.Tensor(max_video_len, embed_size).normal_(0, 0.1))

    def forward(self, x):
        # x (bs=1, seq_len, feature_len)

        bs = x.size(0)
        seq_len = x.size(1)
        feature_len = x.size(2)

        h0 = torch.zeros(2, bs, feature_len).cuda()
        x, _ = self.gru(x, h0)
        x = (x[:, :, :feature_len] + x[:, :, feature_len:])

        x = x.view(seq_len,-1)

        u1 = torch.zeros(seq_len,self.embed_size).cuda()
        for i in range(self.n_segments//2):
            if(i == (self.n_segments//2)-1):
                x_seg = x[(seq_len // (self.n_segments//2)) * i:, :]
            else:
                x_seg = x[(seq_len // (self.n_segments//2)) * i:(seq_len // (self.n_segments//2)) * (i + 1), :]

            q = x_seg
            u_seg = self.dropout(self.A1[i*(self.hops+1)](q))

            # Adjacent weight tying
            for k in range(self.hops):
                m = self.dropout(self.A1[i*(self.hops+1) + k](x_seg))
                if self.temporal_encoding:
                    if (i == (self.n_segments // 2) - 1):
                        m = m + self.TA[(seq_len // (self.n_segments//2)) * i:seq_len, :]
                    else:
                        m = m + self.TA[(seq_len // (self.n_segments//2)) * i:(seq_len // (self.n_segments//2)) * (i + 1), :]

                c = self.dropout(self.A1[i*(self.hops+1) + k + 1](x_seg))
                if self.temporal_encoding:
                    if (i == (self.n_segments // 2) - 1):
                        c = c + self.TC[(seq_len // (self.n_segments//2)) * i:seq_len, :]
                    else:
                        c = c + self.TC[(seq_len // (self.n_segments//2)) * i:(seq_len // (self.n_segments//2)) * (i + 1), :]

                p = torch.matmul(m, u_seg.transpose(1, 0)).transpose(1, 0)
                p = F.softmax(p, -1)
                o = torch.matmul(p, c)
                u_seg = o * u_seg
                u_seg = self.memory_update1[i*self.hops + k](u_seg)
                u_seg = self.layer_norm1[i*self.hops + k](u_seg)

            if (i == (self.n_segments // 2) - 1):
                u1[(seq_len // (self.n_segments//2)) * i:, :] = u_seg
            else:
                u1[(seq_len // (self.n_segments//2)) * i:(seq_len // (self.n_segments//2)) * (i + 1), :] = u_seg

        u2 = torch.zeros(seq_len, self.embed_size).cuda()
        for i in range(self.n_segments // 2):
            x_seg = x[i:seq_len:(self.n_segments//2), :]

            q = x_seg
            u_seg = self.dropout(self.A2[i*(self.hops+1)](q))

            # Adjacent weight tying
            for k in range(self.hops):
                m = self.dropout(self.A2[i*(self.hops+1)+k](x_seg))
                if self.temporal_encoding:
                    m = m + self.TA[i:seq_len:(self.n_segments//2), :]

                c = self.dropout(self.A2[i*(self.hops+1) + k + 1](x_seg))
                if self.temporal_encoding:
                    c = c + self.TC[i:seq_len:(self.n_segments//2), :]

                p = torch.matmul(m, u_seg.transpose(1, 0)).transpose(1, 0)
                p = F.softmax(p, -1)
                o = torch.matmul(p, c)
                u_seg = o * u_seg
                u_seg = self.memory_update2[i*self.hops + k](u_seg)
                u_seg = self.layer_norm2[i*self.hops + k](u_seg)

            u2[i:seq_len:(self.n_segments//2), :] = u_seg

        u = u1 + u2
        y = self.linear_x(x)

        u = u + y
        u = self.layer_norm_y(u)
        u = self.drop50(u)

        # Two layer NN
        u = self.ka(u)
        u = self.layer_norm_ka(u)
        u = self.relu(u)
        u = self.drop50(u)

        out = self.linear(u)
        out = out.view(1, -1)
        out = self.sig(out)

        return out


if __name__ == "__main__":
    input = torch.Tensor(1, 200, 1024).normal_(0, 0.1).cuda()
    print(input.shape)
    print(input)
    model = RUAMN(feature_size=1024, embed_size=512, hops=4, dropout=0.2).cuda()
    out = model(input)
    print(out.shape)
    print(out)