from diff_pool import DiffPool_Prompt, DiffPool
from torch.nn import functional as F
import os
from torch.optim import Adam
import torch
from torch import nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_GNN(epochs, train_loader, val_loader, GNN_model: nn.Module, args):
    device = args.device
    model = GNN_model.to(device)
    optimizer = Adam(model.parameters(), args.lr, weight_decay=5e-4)
    min_loss = 1e10
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model.forward(data)
            if type(model) is DiffPool_Prompt:
                model_type = str(type(model)).split("'")[1].split('.')[1]
                loss1 = F.nll_loss(out[0], data.y.view(-1))
                loss2 = out[1]
                loss3 = F.mse_loss(out[2],
                                   data.graph_degree_distribution.view(data.graph_degree_distribution.shape[0], -1))
                loss = 1.0 * loss1 + 1.0 * loss2 + loss3
                print(f'epoch:{epoch}_batch:{i}_loss:{loss}_loss1:{loss1}_loss2:{loss2}_loss3:{loss3}')
            if type(model) is DiffPool:
                model_type = str(type(model)).split("'")[1]
                loss = F.nll_loss(out, data.y.view(-1))
                print(f'epoch:{epoch}_batch:{i}_loss:{loss}')
            loss.backward()
            optimizer.step()
        val_acc, val_loss = evaluate(args, model, val_loader)
        print('****************val loss:', val_loss, ' val accuracy:', val_acc, '*******************')
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
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print('Ignoring "' + str(e) + '"')
    return model


def evaluate(args, model, loader):
    model.eval()
    correct = 0
    loss = 0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        if type(model) is DiffPool_Prompt:
            pred = out[0].max(dim=1)[1]
            loss1 = F.nll_loss(out[0], data.y.view(-1))
            loss2 = out[1]
            loss3 = F.mse_loss(out[2], data.graph_degree_distribution.view(data.graph_degree_distribution.shape[0], -1))
            loss += loss1 + loss2 + loss3
        if type(model) is DiffPool:
            pred = out.max(dim=1)[1]
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


def test_model(args, model, testloader):
    test_acc, test_loss = evaluate(args, model, testloader)
    print('test accuracy:', test_acc, 'test loss:', test_loss)
    return test_acc
