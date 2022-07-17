from torch_geometric.datasets import Planetoid, CitationFull, Coauthor, Amazon
import utils
import torch

seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_data(args):
    if args.dataset == 'Photo':
        PhotoDataSet = Amazon(root='datasets', name='Photo')
        data = PhotoDataSet[0]
        data.num_classes = PhotoDataSet.num_classes

    if args.dataset == 'Computers':
        ComputersDataSet = Amazon(root='datasets', name='Computers')
        data = ComputersDataSet[0]
        data.num_classes = ComputersDataSet.num_classes

    if args.dataset == 'Physics':
        PhysicsDataSet = Coauthor(root='datasets', name='Physics')
        data = PhysicsDataSet[0]
        data.num_classes = PhysicsDataSet.num_classes

    if args.dataset == 'CS':
        CoauthorDataSet = Coauthor(root='datasets', name='CS')
        data = CoauthorDataSet[0]
        data.num_classes = CoauthorDataSet.num_classes

    if args.dataset == 'DBLP':
        DBLPDataSet = CitationFull(root='datasets', name='DBLP').shuffle()
        data = DBLPDataSet[0]
        data.num_classes = DBLPDataSet.num_classes
    if args.dataset == 'Cora':
        CoraDataSet = Planetoid(root='datasets', name='Cora').shuffle()
        data = CoraDataSet[0]
        data.num_classes = CoraDataSet.num_classes
    if args.dataset == 'Citeseer':
        CiteseerDataSet = Planetoid(root='datasets', name='Citeseer').shuffle()
        data = CiteseerDataSet[0]
        data.num_classes = CiteseerDataSet.num_classes
    if args.dataset == 'PubMed':
        PubMedDataSet = Planetoid(root='datasets', name='Pubmed').shuffle()
        data = PubMedDataSet[0]
        data.num_classes = PubMedDataSet.num_classes

    data = utils.get_node_degree(data=data, edge_type=args.edge_type)
    data = utils.get_neighs_degree_matrix(data=data, edge_type=args.edge_type)
    data = utils.get_degree_sum_neigh(data=data, edge_type=args.edge_type)
    data = utils.set_data_mask(data=data, train_ratio=args.train_ratio, test_ratio=args.test_ratio,
                               pred_ratio=args.pred_ratio)

    return data
