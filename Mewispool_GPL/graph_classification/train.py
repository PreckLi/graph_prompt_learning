import os.path as osp
import time
import torch
import os
from torch_geometric.datasets import TUDataset
from data_deal import TUDataSet_Build
from torch_geometric.loader import DataLoader
from models import Net,Net2,Net3,Net2Prompt,Net3Prompt
from torch.nn.functional import mse_loss


torch.manual_seed(7)
os.makedirs('checkpoints', exist_ok=True)

DATASET_NAME = 'NCI1'
BATCH_SIZE = 20
HIDDEN_DIM = 32
EPOCHS = 200
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.1
EARLY_STOPPING_PATIENCE = 50
k1=1.0
k2=0.5
k3=0.5

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', DATASET_NAME)
# dataset = TUDataset(path, name=DATASET_NAME, use_node_attr=True, use_edge_attr=True).shuffle()
dataset = TUDataSet_Build(path, name=DATASET_NAME, use_node_attr=True, use_edge_attr=True).shuffle()

n = (len(dataset) + 9) // 10

input_dim = dataset.num_features
num_classes = dataset.num_classes

dataset = dataset.shuffle()

test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# device = torch.device('cpu')
device = torch.device('cuda:1')

# Model, Optimizer and Loss definitions
model = Net2(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes).to(device)
# model = Net2Prompt(input_dim=input_dim, hidden_dim=HIDDEN_DIM,distribution_shape=dataset.data.graph_degree_distribution.shape[1],num_classes=num_classes).to(device)
# model = Net3Prompt(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       patience=SCHEDULER_PATIENCE,
                                                       factor=SCHEDULER_FACTOR,
                                                       verbose=True)
print(type(model))
print(DATASET_NAME)
print(device)

nll_loss = torch.nn.NLLLoss()

best_val_loss = float('inf')
best_test_acc = 0
wait = None
for epoch in range(EPOCHS):
    # Training the model
    s_time = time.time()
    train_loss = 0.
    train_corrects = 0
    model.train()
    for i, data in enumerate(train_loader):
        s = time.time()
        data = data.to(device)
        optimizer.zero_grad()

        if type(model) is Net2:
            out, loss_pool = model(data.x, data.edge_index, data.batch)
            loss_classification = nll_loss(out, data.y.view(-1))
            train_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        if type(model) is Net3Prompt:
            loss_classification1 = nll_loss(out[0], data.y.view(-1))
            loss_classification2 = out[1]
            loss_classification3 = mse_loss(out[2], data.graph_degree_distribution)
            loss_classification = loss_classification1 + loss_classification2 + loss_classification3
            train_corrects += out[0].max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        if type(model) is Net2Prompt:
            out, loss_pool = model(data.x, data.edge_index, data.degree, data.batch)
            loss_classification1 = nll_loss(out[0], data.y.view(-1))
            loss_classification2 = out[1]
            loss_classification3 = mse_loss(out[2],data.graph_degree_distribution)
            loss_classification = k1*loss_classification1 + k2*loss_classification2+k3*loss_classification3
            train_corrects += out[0].max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        loss = loss_classification + 0.01 * loss_pool
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        # print(f'{i}/{len(train_loader)}, {time.time() - s}')
    train_loss /= len(train_loader)
    train_acc = train_corrects / len(train_dataset)
    scheduler.step(train_loss)

    # Validation
    val_loss = 0.
    val_corrects = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            s = time.time()
            data = data.to(device)
            if type(model) is Net2:
                out, loss_pool = model(data.x, data.edge_index, data.batch)
                loss_classification = nll_loss(out, data.y.view(-1))
                val_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
            if type(model) is Net3Prompt:
                loss_classification1 = nll_loss(out[0], data.y.view(-1))
                loss_classification2 = out[1]
                loss_classification3 = mse_loss(out[2], data.graph_degree_distribution)
                loss_classification = loss_classification1 + loss_classification2 + loss_classification3
                val_corrects += out[0].max(dim=1)[1].eq(data.y.view(-1)).sum().item()
            if type(model) is Net2Prompt:
                out, loss_pool = model(data.x, data.edge_index, data.degree, data.batch)
                loss_classification1 = nll_loss(out[0], data.y.view(-1))
                loss_classification2 = out[1]
                loss_classification3 = mse_loss(out[2], data.graph_degree_distribution)
                loss_classification = k1*loss_classification1 + k2*loss_classification2+k3*loss_classification3
                val_corrects += out[0].max(dim=1)[1].eq(data.y.view(-1)).sum().item()
            loss = loss_classification + 0.01 * loss_pool
            val_loss += loss.item()

            # print(f'{i}/{len(val_loader)}, {time.time() - s}')

    val_loss /= len(val_loader)
    val_acc = val_corrects / len(val_dataset)

    # Test
    test_loss = 0.
    test_corrects = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            s = time.time()
            data = data.to(device)
            if type(model) is Net2:
                out, loss_pool = model(data.x, data.edge_index, data.batch)
                loss_classification = nll_loss(out, data.y.view(-1))
                test_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
            if type(model) is Net3Prompt:
                loss_classification1 = nll_loss(out[0], data.y.view(-1))
                loss_classification2 = out[1]
                loss_classification3 = mse_loss(out[2], data.graph_degree_distribution)
                loss_classification = loss_classification1 + loss_classification2 + loss_classification3
                test_corrects += out[0].max(dim=1)[1].eq(data.y.view(-1)).sum().item()
            if type(model) is Net2Prompt:
                out, loss_pool = model(data.x, data.edge_index, data.degree, data.batch)
                loss_classification1 = nll_loss(out[0], data.y.view(-1))
                loss_classification2 = out[1]
                loss_classification3 = mse_loss(out[2], data.graph_degree_distribution)
                loss_classification = k1*loss_classification1 + k2*loss_classification2+k3*loss_classification3
                test_corrects += out[0].max(dim=1)[1].eq(data.y.view(-1)).sum().item()
            loss = loss_classification + 0.01 * loss_pool
            test_loss += loss.item()

            # print(f'{i}/{len(val_loader)}, {time.time() - s}')

    test_loss /= len(test_loader)
    test_acc = test_corrects / len(test_dataset)

    elapse_time = time.time() - s_time
    log = '[*] Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.2f}, Val Loss: {:.3f}, ' \
          'Val Acc: {:.2f}, Test Loss: {:.3f}, Test Acc: {:.2f}, Elapsed Time: {:.1f}'\
        .format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, best_test_acc, elapse_time)
    print(log)

    # Early-Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_test_acc = test_acc
        wait = 0
        # saving the model with best validation loss
        torch.save(model.state_dict(), f'checkpoints/{DATASET_NAME}.pkl')
        print('model has been saved at epoch{}'.format(epoch))
    else:
        wait += 1
    # early stopping
    if wait == EARLY_STOPPING_PATIENCE:
        print('======== Early stopping! ========')
        break

