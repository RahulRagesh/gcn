from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from itertools import product
from gcn.utils import *
from gcn.models import GCN, MLP
import sys 
import ast 
from shutil import copyfile

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


best_model_path = 'results/gcn/cora'

configs = pkl_load(best_model_path+'/configs_dict.pkl')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(configs['dataset'])

# Some preprocessing
features = preprocess_features(features)
if configs['model'] == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif configs['model'] == 'gcn_cheby':
    support = chebyshev_polynomials(adj, configs['max_degree'])
    num_supports = 1 + configs[max_degree]
    model_func = GCN
elif configs['model'] == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(configs['model']))


# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, configs, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
best_val = 0.0

# Train model
for epoch in range(configs['epochs']):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: configs['dropout']})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if acc > best_val:
        best_val  = acc 
        model.save(best_model_path+'/best_model', sess)

    if epoch > configs['early_stopping'] and cost_val[-1] > np.mean(cost_val[-(configs['early_stopping']+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
tf.reset_default_graph()
model.load(best_model_path, sess)

test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
