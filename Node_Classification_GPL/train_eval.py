from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
import os
from torch.optim import Adam
import torch
from torch import nn
from model import GNN, GNN_Origin


def Pretrain_GNN(epochs, data, GNN_model: nn.Module, gnn_type, args=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GNN_model.to(device)
    data = data.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr_1, weight_decay=args.weight_decay_1)
    model.train()
    max_val_acc = -1
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model.forward(data, gnn_type)

        if type(model) is GNN:
            loss1 = F.nll_loss(out[0][data.train_mask], data.y[data.train_mask])
            loss2 = F.mse_loss(out[1][data.train_mask], data.degree_log[data.train_mask])
            loss3 = F.mse_loss(out[2], data.degree_exist_matrix)
            if gnn_type == 'gcn':
                loss = 0.1 * loss1 + 3 * torch.log2(1 + loss2) + 3 * torch.log(1 + loss3)
            if gnn_type == 'gat':
                loss = 0.1 * loss1 + (2.0 * torch.log2(1 + loss2)) / (epoch / 100) + (2.0 * torch.log(1 + loss3)) / (
                        epoch / 100)
            if gnn_type == 'graphsage':
                loss = 0.1 * loss1 + 2 * torch.log2(1 + loss2) + 2 * torch.log(1 + loss3)
            if gnn_type == 'graphconv':
                loss = torch.log2(1 + loss2) + torch.log(1 + loss3)
            if gnn_type == 'transformerconv':
                loss = 2 * torch.log2(1 + loss2) + 2 * torch.log(1 + loss3)
            if gnn_type == 'lgconv':
                loss = 2 * torch.log2(1 + loss2) + 2 * torch.log(1 + loss3)
            if gnn_type == 'ARMAconv':
                loss = 2 * torch.log2(1 + loss2) + 2 * torch.log(1 + loss3)
            if gnn_type == "Chebconv":
                loss = 2 * torch.log2(1 + loss2) + 2 * torch.log(1 + loss3)
            if gnn_type == "GeneralConv":
                loss = 2 * torch.log2(1 + loss2) + 2 * torch.log(1 + loss3)
            if gnn_type == "Supergatconv":
                loss = 2 * torch.log2(1 + loss2) + 2 * torch.log(1 + loss3)
        if type(model) is GNN_Origin:
            pass
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            if type(model) is GNN:
                print('pre train epoch:{},loss:{},loss2:{},loss3:{}'.format(epoch, loss.item(),
                                                                            2 * torch.log2(1 + loss2).item(),
                                                                            2 * torch.log(1 + loss3).item()))
                val_loss = F.nll_loss(out[0][data.val_mask], data.y[data.val_mask])
                val_correct = (out[0].argmax(dim=1)[data.val_mask] == data.y[data.val_mask]).sum()
                val_acc = int(val_correct) / int(data.val_mask.sum())
                print('val_acc:{}'.format(val_acc))
                if val_acc > max_val_acc:
                    torch.save(model.state_dict(), 'pretrain_latest_great_GNN.pth')
                    print("pretrain_model saved at epoch_", epoch)
                    max_val_acc = val_acc
            if type(model) is GNN_Origin:
                pass
    model.load_state_dict(torch.load('pretrain_latest_great_GNN.pth'))
    return model


def train_GNN(epochs, data, GNN_model: nn.Module, gnn_type, args=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GNN_model.to(device)
    data = data.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr_2, weight_decay=args.weight_decay_2)

    model.train()
    max_val_acc = -1
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model.forward(data, gnn_type)

        if type(model) is GNN:
            loss1 = F.nll_loss(out[0][data.train_mask], data.y[data.train_mask])
            loss2 = F.mse_loss(out[1][data.train_mask], data.degree_log[data.train_mask])
            loss3 = F.mse_loss(out[2], data.degree_exist_matrix)
            if gnn_type == 'gcn':
                loss = 1.0 * loss1 + 1.0 * torch.log2(1 + loss2) + 1.0 * torch.log(1 + loss3)

            if gnn_type == 'gat':
                loss = 1.07 * loss1 + 1.2 * torch.log2(1 + loss2) + 1.0 * torch.log(1 + loss3)

            if gnn_type == 'graphsage':
                loss = 1.45 * loss1 + 0.65 * torch.log2(1 + loss2) + 1.1 * torch.log(1 + loss3)

            if gnn_type == 'transformerconv':
                loss = 0.5 * loss1 + 1.75 * torch.log2(1 + loss2) + 1.5 * torch.log(1 + loss3)

            if gnn_type == 'lgconv':
                loss = 1.0 * loss1 + 1.0 * torch.log2(1 + loss2) + 1.0 * torch.log(1 + loss3)

            if gnn_type == 'ARMAconv':
                loss = 1.0 * loss1 + 1 * torch.log2(1 + loss2) + 1.0 * torch.log(1 + loss3)

            if gnn_type == "Chebconv":
                loss = 0.5 * loss1 + 1.5 * torch.log2(1 + loss2) + 0.5 * torch.log(1 + loss3)

            if gnn_type == "GeneralConv":
                loss = 1.0 * loss1 + 1.0 * torch.log2(1 + loss2) + 1.0 * torch.log(1 + loss3)

            if gnn_type == "Supergatconv":
                loss = 1.5 * loss1 + 1.0 * torch.log2(1 + loss2) + 1.0 * torch.log(1 + loss3)

        if type(model) is GNN_Origin:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            if type(model) is GNN:
                print('train epoch:{},loss:{},loss1:{},loss2:{},loss3:{}'.format(epoch, loss.item(), loss1,
                                                                                 torch.log2(1 + loss2).item(),
                                                                                 torch.log(1 + loss3).item()))
                val_loss = F.nll_loss(out[0][data.val_mask], data.y[data.val_mask])
                val_correct = (out[0].argmax(dim=1)[data.val_mask] == data.y[data.val_mask]).sum()
                val_acc = int(val_correct) / int(data.val_mask.sum())
                print('val_acc:{}'.format(val_acc))
                if val_acc > max_val_acc:
                    torch.save(model.state_dict(), 'train_latest_great_GNN.pth')
                    print("train_model saved at epoch_", epoch)
                    max_val_acc = val_acc
            if type(model) is GNN_Origin:
                print('train epoch:{},loss:{}'.format(epoch, loss))

    if type(model) is GNN:
        model.load_state_dict(torch.load('train_latest_great_GNN.pth'))
    if type(model) is GNN_Origin:
        pass
    return model


def evaluate_GNN(data, GNN_model, gnn_style):
    model = GNN_model
    model.eval()

    pred = model(data, gnn_style)
    if type(model) is GNN_Origin:
        pred = pred.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print('label accuracy:{}'.format(acc))

    if type(model) is GNN:
        label_correct = (pred[0].argmax(dim=1)[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(label_correct) / int(data.test_mask.sum())
        print('label accuracy:{}'.format(acc))

    return acc
