from torch.nn import functional as F
import os
from torch.optim import Adam
import torch
from torch import nn
from model import GNN, GNN_Origin

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_GNN(epochs, train_loader, val_loader, GNN_model: nn.Module, args):
    device = torch.device(args.device)
    args.device = device
    model = GNN_model.to(device)
    optimizer1 = Adam(model.parameters(), args.lr, weight_decay=5e-4)
    min_loss = 1e10
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer1.zero_grad()
            out = model.forward(data, args.gnn_type)
            if type(model) is GNN:
                model_type = str(type(model)).split("'")[1].split('.')[1]
                loss1 = F.nll_loss(out[0], data.y)
                loss2 = out[1]
                loss3 = F.mse_loss(out[2], data.graph_degree_distribution)
                loss =args.k1*loss1 + args.k2*loss2 +args.k3*loss3.item()

                print(f'epoch:{epoch}_batch:{i}_loss:{loss}_loss1:{loss1}_loss2:{loss2}_loss3:{loss3}')
            else:
                model_type = str(type(model)).split("'")[1]
                loss = F.nll_loss(out, data.y)
                print(f'epoch:{epoch}_batch:{i}_loss:{loss}')
            loss.backward()
            optimizer1.step()
        val_acc, val_loss = evaluate(args, model, val_loader)
        print('****************val loss:', val_loss, ' val accuracy:', val_acc,'*******************')
        if val_loss < min_loss:
            torch.save(model.state_dict(), f'latest_{args.gnn_type}_{model_type}.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break

    state = torch.load(f'latest_{args.gnn_type}_{model_type}.pth')
    model.load_state_dict(state)
    return model


def evaluate(args, model, loader):
    model.eval()
    correct = 0
    loss = 0
    for data in loader:
        data = data.to(args.device)
        out = model(data, args.gnn_type)
        if type(model) is GNN:
            pred = out[0].max(dim=1)[1]
            loss1 = F.nll_loss(out[0], data.y, reduction='sum').item()
            loss2 = out[1]
            loss3 = F.mse_loss(out[2], data.graph_degree_distribution)
            loss +=args.k1*loss1 + args.k2*loss2 +args.k3*loss3.item()
        else:
            pred = out.max(dim=1)[1]
            loss += F.nll_loss(out, data.y, reduction='sum').item()
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


def test_model(args, model, testloader):
    test_acc, test_loss = evaluate(args, model, testloader)
    print('test accuracy:', test_acc, 'test loss:', test_loss)
    return test_acc