import dgl
import torch
import torch.nn as nn
from math import sqrt
from dgl.nn.pytorch.conv import GraphConv
from utils import topk, get_batch_id

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        self.norm = nn.LayerNorm(dim_v)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear_q.reset_parameters()
        self.linear_k.reset_parameters()
        self.linear_v.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # (batch, nh, n, n)
        dist = torch.softmax(dist, dim=-1)

        att = torch.matmul(dist, v)  # (batch, nh, n, dv)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # (batch, n, dim_v)
        att_n = self.norm(att)
        return att_n

class Pool(torch.nn.Module):
    def __init__(self, in_dim: int, ratio=0.5, conv_op=GraphConv, non_linearity=torch.tanh):
        super(Pool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer1 = conv_op(in_dim, 1)
        self.score_layer2 = nn.Linear(in_dim, 1)
        self.non_linearity = non_linearity
        self.reset_parameters()

    def reset_parameters(self):
        self.score_layer1.reset_parameters()
        self.score_layer2.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, feature: torch.Tensor):
        score1 = self.score_layer1(graph, feature).squeeze()
        score2 = self.score_layer2(feature).squeeze()
        score = torch.max(torch.cat((score1.unsqueeze(1), score2.unsqueeze(1)), dim=1), dim=1)[0]
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()),
                                          graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)
        graph.set_batch_num_nodes(next_batch_num_nodes)

        return graph, feature, perm



