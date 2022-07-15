import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from torch_geometric.utils import degree


def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # 获取节点数
    num_nodes = labels.shape[0]

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # 转为pyg需要的coo形式
    degree, degree_log = get_node_degree(adj)
    node_neighs_degree_sum, node_neighs_degree_sum_log = get_node_neighs_degree_sum(adj, num_nodes)
    degree_max=int(max(degree).numpy())
    degree_exist_matrix=get_neighs_degree_matrix(adj,num_nodes,degree_max)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, degree,degree_log, degree_exist_matrix


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_node_degree(adj):
    edge_index = sp.coo_matrix(adj)
    indices = np.vstack((edge_index.row, edge_index.col))
    edge_index = torch.LongTensor(indices)
    node_degree = degree(edge_index[0])
    node_degree_log = torch.log(node_degree + 1) / torch.log(node_degree.max())
    return node_degree, node_degree_log


def get_node_neighs_degree_sum(adj, num_nodes):
    edge_index = sp.coo_matrix(adj)
    indices = np.vstack((edge_index.row, edge_index.col))
    edge_index = torch.LongTensor(indices)
    degree_dict = get_degree_dict(edge_index)
    neighs_dict = get_node_neighs(edge_index)
    degree_sum_list = list()
    for i in neighs_dict:
        neighs_list = neighs_dict[i]
        temp_sum = 0
        for j in neighs_list:
            temp_sum += degree_dict[j]
        degree_sum_list.append(temp_sum)
    # 若有孤立结点，则这里将孤立结点的degree_sum设为0
    node_list = list(range(num_nodes))
    for i in node_list:
        if i not in edge_index[0]:
            degree_sum_list.insert(i, 0)

    degree_sum = torch.as_tensor(degree_sum_list, dtype=torch.float32)
    # degree_sum_max = degree_sum.max()
    # degree_neighs_sum = degree_sum
    degree_sum_log = np.log(degree_sum + 1) / np.log(degree_sum.max())
    #  get the degree_neighs_sum and degree_neighs_sum_classes
    # degree_neighs_sum_log = degree_sum
    # degree_neighs_sum_classes = len(np.unique(degree_sum_list))
    return degree_sum, degree_sum_log


def get_degree_dict(edge_index):
    """
    get a degree dictionary object of each node in data
    :return: a degree dictionary object
    """
    edge_list = edge_index[0].tolist()
    result = pd.value_counts(edge_list)
    degree_dict = result.to_dict()
    return degree_dict


def get_node_neighs(edge_index):
    """
        get neighborhoods of each node
        :return: a dictionary object of nodes' neighborhoods
        """
    neighs_dict = dict()
    edge_list = edge_index.tolist()
    for i in range(len(edge_list[0])):
        if not neighs_dict.get(edge_list[0][i]):
            neighs_dict[edge_list[0][i]] = [edge_list[1][i]]
        else:
            neighs_dict[edge_list[0][i]].append(edge_list[1][i])
    return neighs_dict

def get_neighs_degree_matrix(adj,num_nodes,degree_max):
    edge_index = sp.coo_matrix(adj)
    indices = np.vstack((edge_index.row, edge_index.col))
    edge_index = torch.LongTensor(indices)
    degree_dict = get_degree_dict(edge_index)
    neighs_dict = get_node_neighs(edge_index)
    degree_exist_matrix = np.zeros((num_nodes, degree_max))
    for i in neighs_dict:
        neighs_list = neighs_dict[i]
        for j in neighs_list:
            degree_exist_matrix[i][degree_dict[j] - 1] += 1
    # x_min, x_max = degree_exist_matrix.min(0), degree_exist_matrix.max(0)
    # x = (degree_exist_matrix - x_min) / (x_max - x_min)
    # x[np.isnan(x)]=0
    x = np.log(degree_exist_matrix + 1) / np.log(degree_max)
    degree_exist_matrix = torch.as_tensor(x, dtype=torch.float32)
    return degree_exist_matrix