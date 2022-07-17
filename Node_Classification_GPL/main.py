from train_eval import train_GNN, evaluate_GNN, Pretrain_GNN
from model import GNN, GNN_Origin
from load_dataset import load_data
import argparse
import numpy as np
import torch
import random
import os

import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument('--edge_type', type=str, default='repeat',
                    help='decide whether the edge type is repeat or not repeat')
parser.add_argument('--train_ratio', type=float, default=0.2, help='train ratio')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')
parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
parser.add_argument('--gnn_type', type=str, default="gcn", help='gnn_type')
parser.add_argument('--lr_1', type=float, default=0.00025, help='Pretrain_learning rate')
parser.add_argument('--weight_decay_1', type=float, default=5e-4, help='Pretrain_weighted decay')
parser.add_argument('--lr_2', type=float, default=0.0075, help='Train_learning rate')
parser.add_argument('--weight_decay_2', type=float, default=5e-4, help='Train_weighted decay')
parser.add_argument('--model_type', type=str, default="prompt", help='prompt or origin')
parser.add_argument('--seed', type=int, default=40, help='seed')
args = parser.parse_args()


def main():
    data = load_data(args)
    pre_epochs = 200
    epochs = 400
    acc_list = []
    logger = get_logger(f'./{args.gnn_type}_{args.dataset}_{args.model_type}_exp.log')
    logger.info('start training!')
    logger.info(f"gnn_type:{args.gnn_type}, dataset:{args.dataset} , model_type:{args.model_type},")
    for i in range(1):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        print("range:{}----------------------------------------------------------".format(i + 1))
        if args.model_type == 'origin':
            model = GNN_Origin(data, hid_size=16, gnn_style=args.gnn_type)
        if args.model_type == 'prompt':
            model = GNN(data, hid_size=16, gnn_style=args.gnn_type)
            model = Pretrain_GNN(pre_epochs, data, model, args.gnn_type, args)  # Pretrain Open
        model = train_GNN(epochs, data, model, args.gnn_type, args)
        acc = evaluate_GNN(data, model, args.gnn_type)
        logger.info(f'Number{i}:acc={acc}')
        acc_list.append(acc)

    mean_acc = np.mean(acc_list)
    max_acc = np.max(acc_list)
    min_acc = np.min(acc_list)
    logger.info(f"mean:{mean_acc}, max:{max_acc}, min:{min_acc}")
    logger.info("finish training")

    print(acc_list)
    print("mean: ", mean_acc)
    print("max: ", max_acc)
    print("min: ", min_acc)

    print('---------args-----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------end!----------\n')


if __name__ == '__main__':
    main()
