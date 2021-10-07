import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d
import dgl
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
from math import ceil
from layer import MultiHeadSelfAttention, Pool
from utils import get_batch_id, to_batch

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.dataname = args.dataname
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.max_num_nodes = args.max_num_nodes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.gnn1 = args.gnn1
        self.gnn2 = args.gnn2
        self.gnn3 = args.gnn3
        self.conv_channel1 = args.conv_channel1
        self.conv_channel2 = args.conv_channel2
        self.dims = ceil(self.max_num_nodes * self.pooling_ratio)

        gnn_type = [self.gnn1, self.gnn2, self.gnn3]
        gconvs = []
        for i in range(len(gnn_type)):
            in_dim = self.num_features if i == 0 else self.nhid
            out_dim = self.nhid
            gconvs.append(self.gnn_selection(gnn_type[i], in_dim, out_dim))
        self.gconvs = torch.nn.ModuleList(gconvs)

        self.pool = Pool(self.nhid, ratio=self.pooling_ratio)

        if self.dataname == 'DD':
            self.norm = nn.LayerNorm(self.nhid)
        else:
            self.att = MultiHeadSelfAttention(self.nhid, self.nhid, self.nhid, 8)

        self.conv1 = Conv2d(3, self.conv_channel1, (self.dims,1))
        self.maxpool2d = MaxPool2d((1, 2), (1, 2))
        self.conv2 = Conv2d(self.conv_channel1, self.conv_channel2, (1, self.nhid //2), 1)

        self.lin1 = nn.Linear(self.conv_channel2, self.conv_channel2)
        self.lin2 = nn.Linear(self.conv_channel2, self.num_classes)

    def reset_parameters(self):
        for gconv in self.gconvs:
            gconv.reset_parameters()
        self.pool.reset_parameters()
        if self.dataname == 'DD':
            self.norm.reset_parameters()
        else:
            self.att.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def gnn_selection(self, gnn, in_channels, out_channels):
        if (gnn == 'GraphConv'):
            return GraphConv(in_channels, out_channels)
        elif (gnn == 'SAGEConv'):
            return SAGEConv(in_channels, out_channels, aggregator_type='mean')
        elif (gnn == 'GATConv'):
            return GATConv(in_channels, out_channels, num_heads=1)

    def forward(self, graph:dgl.DGLGraph):
        feat = graph.ndata["feat"]
        batch_size = graph.batch_size
        batch_x = []
        for i in range(3):
            feat = feat if i ==0 else x
            x = F.relu(self.gconvs[i](graph, feat) if not isinstance(self.gconvs[i], GATConv) else self.gconvs[i](graph, feat).flatten(1))
            graphp, xp, _ = self.pool(graph, x)
            batchp = get_batch_id(graphp.batch_num_nodes())
            batch_xp, _ = to_batch(xp, graphp.batch_num_nodes(), batchp, batch_size, 0, self.dims)
            if self.dataname == 'DD':
                batch_out = self.norm(batch_xp)
            else:
                batch_out = self.att(batch_xp)
            batch_x.append(batch_out.unsqueeze(1))

        x = torch.cat((batch_x), 1)

        x = F.relu(self.conv1(x))
        x = self.maxpool2d(x)
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x
