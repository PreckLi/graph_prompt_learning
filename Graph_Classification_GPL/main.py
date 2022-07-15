import argparse
import random

import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import time

from data_deal import TUDataSet_Build
from model import GNN, GNN_Origin
from train_eval import test_model, train_GNN

import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument('--train_ratio', type=float, default=0.2, help='train ratio')
parser.add_argument('--pred_ratio', type=float, default=0.2, help='predict ratio')
parser.add_argument('--device', type=str, default='cuda:1', help='cuda0 or cuda1')
parser.add_argument('--dataset', type=str, default='NCI1', help='dataset')
parser.add_argument('--gnn_type', type=str, default='sortpool', help='dataset')
parser.add_argument('--model_type', type=str, default="origin", help='model type')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--seed', type=int, default=40, help='seed')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=400, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=1000, help='patience for early stopping')
parser.add_argument('--decomposition_type', type=str, default='tsne', help='tsne/pca')
parser.add_argument('--k1', type=float, default=1.5, help='k1')
parser.add_argument('--k2', type=float, default=0.7, help='k2')
parser.add_argument('--k3', type=float, default=1.0, help='k3')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = TUDataSet_Build(root='datasets', name=args.dataset)
dataset.data.num_classes = dataset.num_classes

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_eval = len(dataset) - num_training - num_val

training_set, validation_set, evaluation_set = random_split(dataset, [num_training, num_val, num_eval])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
eval_loader = DataLoader(evaluation_set, batch_size=args.batch_size, shuffle=False)


def main():
    acc_list = []
    since = time.time()
    logger = get_logger(f'./{args.gnn_type}_{args.dataset}_{args.model_type}.log')
    logger.info('start training!')
    logger.info(
        f"gnn_type:{args.gnn_type}, dataset:{args.dataset}, model_type:{args.model_type}, k1:{args.k1}, k2:{args.k2}, k3:{args.k3}")
    for i in range(5):
        print("range:{}----------------------------------------------------------".format(i + 1))
        if args.model_type == 'prompt':
            model = GNN(dataset.data, args.nhid, gnn_style=args.gnn_type)
        else:
            model = GNN_Origin(dataset.data, args.nhid, gnn_style=args.gnn_type)
        model = train_GNN(args.epochs, train_loader, val_loader, model, args)
        acc = test_model(args, model, eval_loader)
        acc_list.append(acc)
        logger.info(f"NO{i}, acc:{acc}")
    mean_acc = np.mean(acc_list)
    max_acc = max(acc_list)
    min_acc = min(acc_list)
    logger.info(f"mean:{mean_acc}, max:{max_acc}, min:{min_acc}")
    logger.info("Finish")
    print(acc_list)
    print('mean:', mean_acc)
    print('max:', max_acc)
    print('min:', min_acc)
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('---------args-----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------end!----------\n')


if __name__ == '__main__':
    main()
