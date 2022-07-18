import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from data_deal import TUDataSet_Build
from networks import Net, NetPrompt

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/MUTAG')
parser.add_argument('--epochs', type=int, default=400,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--model_type', type=str, default='prompt',
                    help='model_type')
parser.add_argument('--k1', type=float, default=4, help='k1')
parser.add_argument('--k2', type=float, default=0.5, help='k2')
parser.add_argument('--k3', type=float, default=1, help='k3')


def test(model, loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        if args.model_type == 'origin':
            pred = out.max(dim=1)[1]
            loss += F.nll_loss(out, data.y, reduction='sum').item()
        if args.model_type == 'prompt':
            pred = out[0].max(dim=1)[1]
            loss1 = F.nll_loss(out[0], data.y, reduction='sum').item()
            loss2 = out[1]
            loss3 = F.mse_loss(out[2], data.graph_degree_distribution)
            loss += loss1 + torch.log(1 + loss2) + loss3.item()
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


args = parser.parse_args()
args.device = 'cuda:0'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
dataset = TUDataSet_Build(root=os.path.join('data'), name=args.dataset)
distribution_num = dataset.data.graph_degree_distribution.shape[1]
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

acc_list = []
for i in range(5):
    print(f'------------------------range:{i}--------------------------')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_type == 'origin':
        model = Net(args).to(args.device)
    if args.model_type == 'prompt':
        model = NetPrompt(args, distribution_num).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss = 1e10
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data)
            if args.model_type == 'origin':
                loss = F.nll_loss(out, data.y)
            if args.model_type == 'prompt':
                loss1 = F.nll_loss(out[0], data.y)
                loss2 = out[1]
                loss3 = F.mse_loss(out[2], data.graph_degree_distribution)
                loss = args.k1 * loss1 + args.k2 * torch.log(1 + loss2) + args.k3 * loss3.item()
            print("epoch:{},batch:{},Training loss:{}".format(epoch, i, loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_acc, val_loss = test(model, val_loader)
        print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), 'latest.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break

    if args.model_type == 'origin':
        model = Net(args).to(args.device)
    if args.model_type == 'prompt':
        model = NetPrompt(args, distribution_num).to(args.device)
    model.load_state_dict(torch.load('latest.pth'))
    test_acc, test_loss = test(model, test_loader)
    print("Test accuarcy:{}".format(test_acc))
    acc_list.append(test_acc)

print(f"mean: {np.mean(acc_list)}")
print(f"max: {max(acc_list)}")
print(f"min: {min(acc_list)}")

print('---------args-----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------start!----------\n')
