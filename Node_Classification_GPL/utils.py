import torch
import pandas as pd
import numpy as np
import torch_geometric

def get_node_neighs(data: torch_geometric.data.data.Data, edge_type='repeat'):
    """
    get neighborhoods of each node
    :param data:
    :return: a dictionary object of nodes' neighborhoods
    """
    neighs_dict = dict()
    edge_list = data.edge_index.tolist()
    if edge_type == 'repeat':
        for i in range(len(edge_list[0])):
            if not neighs_dict.get(edge_list[0][i]):
                neighs_dict[edge_list[0][i]] = [edge_list[1][i]]
            else:
                neighs_dict[edge_list[0][i]].append(edge_list[1][i])
    else:
        for i in range(len(edge_list[0])):
            if not neighs_dict.get(edge_list[0][i]):
                neighs_dict[edge_list[0][i]] = [edge_list[1][i]]
            else:
                neighs_dict[edge_list[0][i]].append(edge_list[1][i])
            if not neighs_dict.get(edge_list[1][i]):
                neighs_dict[edge_list[1][i]] = [edge_list[0][i]]
            else:
                neighs_dict[edge_list[1][i]].append(edge_list[0][i])
    return neighs_dict


def get_degree_dict(data: torch_geometric.data.data.Data, edge_type):
    """
    get a degree dictionary object of each node in data
    :param data:data of pyg
    :return: a degree dictionary object
    """
    if edge_type == 'repeat':
        edge_list = data.edge_index[0].tolist()
        result = pd.value_counts(edge_list)
    else:
        edge_list = data.edge_index[0].tolist()
        edge_list_extend = data.edge_index[1].tolist()
        edge_list.extend(edge_list_extend)
        result = pd.value_counts(edge_list)
    degree_dict = result.to_dict()
    return degree_dict


def get_node_degree(data: torch_geometric.data.data.Data, edge_type):
    """
    get the degree of each node
    :param data: data of pyg
    :return: the data of pyg which has got the degree of each node
    """
    if edge_type == 'repeat':
        edge_list = data.edge_index[0].tolist()
    else:
        edge_list = data.edge_index[0].tolist()
        edge_list_extend = data.edge_index[1].tolist()
        edge_list.extend(edge_list_extend)
    result = pd.value_counts(edge_list)
    node_list = list(range(data.num_nodes))
    for i in node_list:
        if i not in data.edge_index[0]:
            result[i] = 0

    degree = result.sort_index()
    degree_log = np.log(degree + 1) / np.log(degree.max())
    data.degree_max = degree.max()
    data.degree = torch.as_tensor(degree.to_numpy(), dtype=torch.float32)
    data.degree_log = torch.as_tensor(degree_log.to_numpy(), dtype=torch.float32)
    data.degree_classes = len(np.unique(result))
    return data


def get_degree_sum_neigh(data: torch_geometric.data.data.Data, edge_type):
    """
    get the sum degree of each node's neighborhoods
    :param data: data of pyg
    :return: the data of pyg which has got the sum degree of each node's neighborhood
    """
    degree_dict = get_degree_dict(data, edge_type=edge_type)
    neighs_dict = get_node_neighs(data, edge_type=edge_type)
    degree_sum_list = list()
    for i in neighs_dict:
        neighs_list = neighs_dict[i]
        temp_sum = 0
        for j in neighs_list:
            temp_sum += degree_dict[j]
        degree_sum_list.append(temp_sum)
    node_list = list(range(data.num_nodes))
    for i in node_list:
        if i not in data.edge_index[0]:
            degree_sum_list.insert(i, 0)

    degree_sum = torch.as_tensor(degree_sum_list, dtype=torch.float32)
    data.degree_sum_max = degree_sum.max()
    data.degree_neighs_sum = degree_sum
    degree_sum = np.log(degree_sum + 1) / np.log(degree_sum.max())
    data.degree_neighs_sum_log = degree_sum
    data.degree_neighs_sum_classes = len(np.unique(degree_sum_list))
    return data


def get_neighs_degree_matrix(data: torch_geometric.data.data.Data, edge_type):
    """
    return a matrix which record exist degree of a node,size is (num_nodes,degree_max)
    :param data:
    :param edge_type:
    :return: data
    """
    degree_dict = get_degree_dict(data, edge_type=edge_type)
    neighs_dict = get_node_neighs(data, edge_type=edge_type)
    degree_exist_matrix = np.zeros((data.num_nodes, data.degree_max))
    for i in neighs_dict:
        neighs_list = neighs_dict[i]
        for j in neighs_list:
            degree_exist_matrix[i][degree_dict[j] - 1] += 1
    x_min, x_max = degree_exist_matrix.min(0), degree_exist_matrix.max(0)
    x = (degree_exist_matrix - x_min) / (x_max - x_min)
    x[np.isnan(x)]=0
    degree_exist_matrix = torch.as_tensor(x, dtype=torch.float32)
    data.degree_exist_matrix = degree_exist_matrix
    return data


def set_data_mask(data: torch_geometric.data.data.Data, train_ratio=0.8,test_ratio=0.1, pred_ratio=0.01):
    """
    set the mask for train and evaluation
    :param data: data
    :param train_ratio:train ratio
    :return: data
    """
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    train_mask[:int(train_ratio * data.y.size(0))] = True

    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask[int(train_ratio * data.y.size(0)):int((train_ratio+test_ratio) * data.y.size(0))] = True

    val_mask=torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask[int((train_ratio+test_ratio) * data.y.size(0)):]=True

    pred_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    pred_mask[int((1 - pred_ratio) * data.y.size(0)):] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    data.pred_mask = pred_mask

    return data


def concat_x(data: torch_geometric.data.data.Data):
    """
    concat x with degree
    :param data:
    :return:
    """
    data.x = torch.cat((data.x, data.degree.view(-1, 1)), 1)
    return data