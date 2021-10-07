import argparse
import dgl
from dgl.data import LegacyTUDataset
import torch
import os
import numpy as np
import random
from network import Net
from train_eval import cross_validation_with_val_set

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=8971, help='seed')
parser.add_argument('--folds', type=int, default=10, help='Cross validation folds')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.6, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.3, help='dropout ratio')
parser.add_argument('--gnn1', type=str, default='SAGEConv', help='gnn1 layer type')
parser.add_argument('--gnn2', type=str, default='GraphConv', help='gnn2 layer type')
parser.add_argument('--gnn3', type=str, default='GATConv', help='gnn3 layer type')
parser.add_argument('--convpooling_ratio1', type=int, default=16, help='conv_channel1')
parser.add_argument('--conv_channel2', type=int, default=16, help='conv_channel2')
parser.add_argument('--dataname', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/FRANKENSTEIN')
parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv', help='pooling layer type')
parser.add_argument('--dataset_path', type=str, default='../data', help='path to save result')
args = parser.parse_args()

args.device = 'cpu'
if torch.cuda.is_available():
    if args.dataname == 'PROTEINS':
        args.device = 'cuda:0'
    elif args.dataname == 'DD':
        args.device = 'cuda:0'
    elif args.dataname == 'NCI1':
        args.device = 'cuda:2'
    elif args.dataname == 'NCI109':
        args.device = 'cuda:3'
    elif args.dataname == 'Mutagenicity':
        args.device = 'cuda:2'

if not os.path.exists(args.dataset_path):
    os.makedirs(args.dataset_path)

def setseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    train_loss, train_acc = info['train_loss'], info['train_acc']
    val_loss, val_acc = info['val_loss'], info['val_acc']
    test_loss, test_acc = info['test_loss'], info['test_acc']
    print('{:02d}/{:03d}: train loss: {:.6f}, val loss: {:.6f}, test loss: {:.6f}, train acc: {:.6f}, val acc: {:.6f}, test acc: {:.6f}'.format(
        fold, epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc))

dataset = LegacyTUDataset(args.dataname, raw_dir=args.dataset_path)
for i in range(len(dataset)):
    dataset.graph_lists[i] = dgl.add_self_loop(dataset.graph_lists[i])
num_features, num_classes, max_num_nodes = dataset.statistics()
args.num_features = int(num_features)
args.num_classes = int(num_classes)
args.max_num_nodes = int(max_num_nodes)

setseed(args.seed)
model = Net(args)
acc, std, duration_mean = cross_validation_with_val_set(
    dataset,
    model,
    seed=args.seed,
    folds=args.folds,
    lr=args.lr,
    weight_decay=args.weight_decay,
    batch_size=args.batch_size,
    epochs=args.epochs,
    device=args.device,
    patience=args.patience,
    logger=logger,
)





