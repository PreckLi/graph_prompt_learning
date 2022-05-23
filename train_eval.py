from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
import os
from torch.optim import Adam
import torch
from torch import nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_GNN(epochs, data, GNN_model: nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN_model.to(device)
    data = data.to(device)
    optimizer1 = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    lf2 = nn.L1Loss()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer1.zero_grad()
        out = model.forward(data)
        loss1 = F.nll_loss(out[0][data.train_mask], data.y[data.train_mask])
        loss2 = lf2(out[1][data.train_mask], data.degree_log[data.train_mask])
        loss3 = F.mse_loss(out[2][data.train_mask], data.degree_neighs_sum_log[data.train_mask])

        loss = loss1 + torch.log2(1 + loss2) + torch.log(1 + loss3)
        loss.backward()
        optimizer1.step()

        if epoch % 50 == 0:
            print('epoch:{},loss:{}'.format(epoch, loss.item()))

    return GNN_model


def evaluate_GNN(data, GNN_model, ratio=0.99):
    model = GNN_model
    model.eval()

    pred = model(data)
    label_correct = (pred[0].argmax(dim=1)[data.test_mask] == data.y[data.test_mask]).sum()
    label_acc = int(label_correct) / int(data.test_mask.sum())

    degree_pred = pred[1][data.pred_mask]
    degree_neighs_pred = pred[2][data.pred_mask]

    x = np.linspace(int(ratio * data.num_nodes), int(data.num_nodes), int(data.num_nodes - ratio * data.num_nodes) + 1)

    degree_pred = torch.round(data.degree_max ** degree_pred - 1)  # 还原成原来度的值，并四舍五入取整
    degree_y_pred = degree_pred.cpu().detach().numpy()
    degree_y_real = data.degree[data.pred_mask].cpu().detach().numpy()
    degree_acc = np.sum(degree_y_pred == degree_y_real) / int(len(degree_y_pred))
    print('Degree accuracy:{}'.format(degree_acc))

    degree_neighs_pred = torch.round(data.degree_sum_max ** degree_neighs_pred - 1)  # 还原成原来邻居度和的值，并四舍五入取整
    degree_neighs_y_pred = degree_neighs_pred.cpu().detach().numpy()
    degree_neighs_y_real = data.degree_neighs_sum[data.pred_mask].cpu().detach().numpy()
    degree_neighs_sum_acc = np.sum(degree_neighs_y_pred == degree_neighs_y_real) / int(len(degree_neighs_y_pred))
    print("Degrees' neighborhoods sum accuracy:{}".format(degree_neighs_sum_acc))

    fig, axs = plt.subplots(2, 1, dpi=800, constrained_layout=True)
    axs[0].plot(x, degree_y_pred, label='pred')
    axs[0].plot(x, degree_y_real, label='real')
    axs[1].plot(x, degree_neighs_y_pred, label='pred')
    axs[1].plot(x, degree_neighs_y_real, label='real')
    axs[0].set_title('Degree Prediction')
    axs[1].set_title("Neighborhoods' Degree Sum Prediction")
    axs[0].set_xlabel('Mask')
    axs[0].set_ylabel('Number Of Degree After Log', fontsize=8)
    axs[1].set_xlabel('Mask')
    axs[1].set_ylabel("Number Of Neighborhoods' Degree After Log", fontsize=6)
    axs[0].legend()
    axs[1].legend()

    plt.show()

    print('label accuracy:{}'.format(label_acc))
