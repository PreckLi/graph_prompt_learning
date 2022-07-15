import argparse
from torch_geometric.loader import DenseDataLoader as DenseLoader
from kernel_datasets import get_dataset
from diff_pool import DiffPool, DiffPool_Prompt
from train_eval import train_GNN, test_model
import random
from torch.utils.data import random_split
import numpy as np
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--dataset', type=str, default='NCI109', help='dataset')
parser.add_argument('--gnn_type', type=str, default='diffpool', help='dataset')
parser.add_argument('--model_type', type=str, default="prompt", help='model type')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--num_layers', type=int, default=2, help='num layers')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=1000, help='patience for early stopping')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = get_dataset(name=args.dataset, sparse=False)
dataset.data.num_classes = dataset.num_classes

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_eval = len(dataset) - num_training - num_val

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_training, num_val, num_eval])
train_loader = DenseLoader(train_dataset, args.batch_size, shuffle=False)
val_loader = DenseLoader(val_dataset, args.batch_size, shuffle=False)
test_loader = DenseLoader(test_dataset, args.batch_size, shuffle=False)

print('---------args-----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------start!----------\n')


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


def main():
    acc_list = []
    since = time.time()
    for i in range(1):
        print("range:{}----------------------------------------------------------".format(i + 1))
        if args.model_type == 'prompt':
            model = DiffPool_Prompt(dataset, args.num_layers, args.nhid)
        else:
            model = DiffPool(dataset, args.num_layers, args.nhid)
        model = train_GNN(args.epochs, train_loader, val_loader, model, args)
        acc = test_model(args, model, test_loader)
        acc_list.append(acc)
    mean_acc = np.mean(acc_list)
    max_acc = max(acc_list)
    min_acc = min(acc_list)
    print(acc_list)
    print('mean:', mean_acc)
    print('max:', max_acc)
    print('min:', min_acc)
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    print('---------args-----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------end!----------\n')


if __name__ == '__main__':
    main()
