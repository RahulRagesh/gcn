import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os
import time
from os import listdir
from os.path import isdir, join

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def create_dir(dir_path):
    """
    Create directory structure given as `dir_path`.
    If path exists ask whether to overwrite it or make a new directory by appending Run ID to it.

    return: path to directory created
    """
        
    if os.path.exists(dir_path):    
        run_id = 0
        dir_path = dir_path+'-%04d'%run_id
        while(os.path.exists(dir_path)):
            run_id = run_id + 1
            dir_path = dir_path[:-4]+'%04d'%run_id

    os.makedirs(dir_path)
    return dir_path


def load_config_file(config_file):
    """
    load `config_file` which contains one parameter in each line of the form `param_name`=`param_value`.
    lines starting with `#` are ignored while parsing.
    """
    config_dict = {}
    with open(config_file, 'r') as f:
        line = f.readline().strip()
        while line:
            if line[0] != '#':
                line = line.replace(' ', '').split('=')
                if len(line) != 2:
                    print('Warning: Broken configuration item!')
                else:
                    config_dict[line[0]] = line[1]
            line = f.readline().strip()
    return config_dict

def update_params_for_config(param_names, params, configs_dict):
    params_str = ""
    for param_name, param in zip(param_names,params):
        target_dict,target_param = param_name.split('.')
        eval(target_dict)[target_param] = param
        param_str = str(param)
        params_str  = "%s\t%s_%s"%(params_str,target_param,param_str)
    return configs_dict, params_str.strip()

def get_hyperparameter_configs(*param_dicts):
    param_names = []
    param_lists = []
    for param_dict in param_dicts:
        for key, param in param_dict.items():
            if type(param) is list and len(param)>1: 
                exec("%s = %s"%(key+"_list",param))
                param_names.append("configs_dict."+key)
                param_lists.append(eval(key+"_list"))
    return param_names,param_lists


# Define model evaluation function
def evaluate(sess, model, features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.pred_error, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

def pkl_dump(file_path, obj_to_dump, mode='wb', protocol=pkl.HIGHEST_PROTOCOL):
    """
    dump `obj_to_dump` to `file_path` with pickle `protocol` (default: pickle.HIGHEST_PROTOCOL)
    """
    pkl.dump(obj_to_dump, open(file_path, mode), protocol)

def pkl_load(file_path, mode='rb'):
    """
    load pickle file from `file_path`
    """
    return pkl.load(open(file_path, mode))

def indices_to_aggregator(indices):
  aggregator_matrix = sp.lil_matrix((indices.shape[0], indices.shape[0]), dtype=np.float32)
  for i, (a,b) in enumerate(indices):
    aggregator_matrix[i,np.argwhere(indices[:,0]==a)[:,0]] = 1
  return sparse_to_tuple(aggregator_matrix)


def row_normalize_sparse_matrix(matrix, gamma = -1):
    b_graph = matrix.astype(np.float32).copy()
    r_graph = matrix.astype(np.float32).copy()

    row_sums = []
    for i in range(matrix.shape[0]):
        row_sum = r_graph.data[r_graph.indptr[i]:r_graph.indptr[i+1]].sum()
        if row_sum == 0:
            row_sums.append(0.0)
        else:
            row_sums.append(row_sum**gamma)

    for i in range(matrix.shape[0]):
        if row_sums[i] != 0:
            b_graph.data[r_graph.indptr[i]:r_graph.indptr[i+1]] *= row_sums[i]    
    
    return b_graph