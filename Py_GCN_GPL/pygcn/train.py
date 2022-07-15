from __future__ import division
from __future__ import print_function

import os
import glob
import time
import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GCN, GCN_prompt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--model_type', type=str, default='prompt',
                    help='model type')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
os.environ["PYTHONHASHSEED"]=str(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


# Load data
adj, features, labels, idx_train, idx_val, idx_test, degree_log, degree_sum_log,degree_exist_matrix = load_data()

# Model and optimizer
if args.model_type == 'origin':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)

if args.model_type == 'prompt':
    model = GCN_prompt(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       degree_max=degree_exist_matrix.shape[1],
                       dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    degree_log = degree_log.cuda()
    degree_sum_log = degree_sum_log.cuda()
    degree_exist_matrix = degree_exist_matrix.cuda()

features, adj, labels, degree_log, degree_sum_log,degree_exist_matrix = Variable(features), Variable(adj), Variable(labels), Variable(
    degree_log), Variable(degree_sum_log),Variable(degree_exist_matrix)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    if args.model_type == 'origin':
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    if args.model_type == 'prompt':
        loss1 = F.nll_loss(output[0][idx_train], labels[idx_train])
        loss2 = F.mse_loss(output[1][idx_train], degree_log[idx_train])
        loss3 = F.mse_loss(output[2][idx_train], degree_exist_matrix[idx_train])
        loss_train = 3*loss1 + 2*torch.log2(1 + loss2) + 1.5*loss3
        acc_train = accuracy(output[0][idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    if args.model_type == 'origin':
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    if args.model_type == 'prompt':
        loss_val = F.nll_loss(output[0][idx_val], labels[idx_val])
        acc_val = accuracy(output[0][idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def test():
    model.eval()
    output = model(features, adj)
    if args.model_type == 'origin':
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    if args.model_type == 'prompt':
        loss_test = F.nll_loss(output[0][idx_test], labels[idx_test])
        acc_test = accuracy(output[0][idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return float(acc_test)


print('---------args-----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------start!----------\n')
acc_list = list()
for i in range(10):
    print('--------------------------------------------range{}----------------------------------------------------'.format(
        i))
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train(epoch))

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    acc_test = test()
    acc_list.append(acc_test)
acc_mean = np.mean(acc_list)
acc_max = max(acc_list)
acc_min = min(acc_list)
print('acc_list:{}'.format(acc_list))
print('acc_mean:{}'.format(acc_mean))
print('acc_max:{}'.format(acc_max))
print('acc_min:{}'.format(acc_min))
