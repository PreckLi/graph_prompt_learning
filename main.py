from data_deal import PlanetoidPubMed
from torch_geometric.datasets import Planetoid
import utils
from train_eval import train_GNN, evaluate_GNN
from model import GNN
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--edge_type',type=str,default='repeat',help='decide whether the edge type is repeat or not repeat')
parser.add_argument('--train_ratio',type=float,default=0.2,help='train ratio')
parser.add_argument('--pred_ratio',type=float,default=0.2,help='predict ratio')
parser.add_argument('--dataset',type=str,default='PubMedDataSet',help='dataset')
parser.add_argument('--gnn_type',type=str,default='gcn',help='dataset')
args=parser.parse_args()

if args.dataset=='PubMedDataSet':
    PubMedDataSet = PlanetoidPubMed(root='datasets/PlantoidPubMed').shuffle()
    data = PubMedDataSet[0]
    data.num_classes = PubMedDataSet.num_classes
if args.dataset=='Cora':
    CoraDataSet=Planetoid(root='datasets',name='Cora').shuffle()
    data=CoraDataSet[0]
    data.num_classes = CoraDataSet.num_classes
if args.dataset=='Citeseer':
    CiteseerDataSet=Planetoid(root='datasets',name='Citeseer').shuffle()
    data=CiteseerDataSet[0]
    data.num_classes = CiteseerDataSet.num_classes


data = utils.get_node_degree(data=data,edge_type=args.edge_type)
# data = utils.get_neighs_degree_matrix(data=data,edge_type=args.edge_type)
# data = utils.concat_x(data=data)
data = utils.get_degree_sum_neigh(data=data,edge_type=args.edge_type)
data = utils.set_data_mask(data=data,train_ratio=args.train_ratio,pred_ratio=args.pred_ratio)

print('---------args-----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------start!----------\n')


def main():
    model = GNN(data, hid_size=16,gnn_style=args.gnn_type)
    epochs = 400
    model = train_GNN(epochs, data, model)
    evaluate_GNN(data, model,ratio=1-args.pred_ratio)


if __name__ == '__main__':
    main()
