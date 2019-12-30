from __future__ import division
from __future__ import print_function

from gcn.utils import *
import scipy.io as io 
import networkx 
import scipy.sparse as sp 
import sys 

def find_connected_components(graph):
    G = nx.from_scipy_sparse_matrix(graph)
    return nx.number_connected_components(G), sorted(nx.connected_components(G), key = len, reverse=True)

dataset = sys.argv[1]
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

print(dataset)
n, comps = find_connected_components(adj)
print(n)
comps = [np.array(list(x)) for x in comps]

m_dict = {}
for i in range(n):
    m_dict['component_'+str(i)] = comps[i]

io.savemat(dataset,mdict=m_dict)

print(np.unique(np.argmax(y_train[train_mask],axis=1),return_counts=True)[1])
print(np.unique(np.argmax(y_val[val_mask],axis=1),return_counts=True)[1])
print(np.unique(np.argmax(y_test[test_mask],axis=1),return_counts=True)[1])