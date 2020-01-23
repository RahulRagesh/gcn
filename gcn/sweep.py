from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from itertools import product
from utils import *
from models import GCN, MLP
import sys 
import ast 
from shutil import copyfile
import pandas as pd 

config_dict = load_config_file(sys.argv[1])
configs = ast.literal_eval(config_dict['configs'])
output_dir = create_dir(configs['output_dir'])
training_params_dict = ast.literal_eval(config_dict['training_params'])
feature_prop_params_dict = ast.literal_eval(config_dict['feature_prop_params'])

try:
    feature_prop_attentive_regs = []
    for l in range(configs['n_layers']):
        feature_prop_attentive_regs.append(np.round(np.linspace(feature_prop_params_dict['attentive_reg'][0][l],feature_prop_params_dict['attentive_reg'][1][l], feature_prop_params_dict['attentive_reg'][-1]),5))

    feature_prop_params_dict['attentive_reg'] = list(product(*feature_prop_attentive_regs))
except:
    pass 

label_prop_params_dict = ast.literal_eval(config_dict['label_prop_params'])
ssl_params_dict = ast.literal_eval(config_dict['ssl_params'])
configs_dict = {**configs, **training_params_dict, **feature_prop_params_dict, **label_prop_params_dict, **ssl_params_dict}

copyfile(sys.argv[1], output_dir+'/config.txt')
training_logs_dir = create_dir(output_dir+'/training_logs')

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

configs_dict['num_indices'] =  support[0][0].shape[0]
configs_dict['num_nodes'] = support[0][2][0]
column_header = ['Epoch', 'Training_Loss', 'Pred_Error', 'Attentive_Reg', 'Weight_Reg', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'Test_Loss', 'Test_Acc']
column_map = dict(list(zip(column_header,range(len(column_header)))))
best_model_ckpt_path = output_dir + '/best_model_checkpoint/'

configs_dict['attentive_feat_prop_init'] = np.reshape(support[0][1], (-1,))
configs_dict['output_dir'] = output_dir

configs_dict['global_aggregator_matrix'] = sparse_to_tuple(row_normalize_sparse_matrix(adj + sp.eye(adj.shape[0])))
configs_dict['attentive_label_prop_init'] = np.reshape(configs_dict['global_aggregator_matrix'][1], (-1,1))

if configs_dict.get('propagate_labels',False):
    if configs_dict.get('learnable_label_propagation',False):
        configs_dict['label_aggregator_matrix'] = indices_to_aggregator(sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0])
    else:
        configs_dict['label_aggregator_matrix'] = sparse_to_tuple(row_normalize_sparse_matrix(adj + sp.eye(adj.shape[0])))

val_logs = []
config_names = []
cost_val = []
best_val_loss = 10000000  

param_names, param_lists = get_hyperparameter_configs(configs_dict)

for config_count, params in enumerate(product(*param_lists)):
    tf.reset_default_graph()    
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    configs_dict, params_str = update_params_for_config(param_names, params, configs_dict)
    
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
    model = model_func(placeholders, configs_dict, input_dim=features[2][1], logging=True)

    with tf.Session() as sess:                
        sess.run(tf.global_variables_initializer())
        # Train model
        patience_count = 0
        training_log = []
        best_config_stats = []
        best_config_val_loss = 10000000
        for epoch in range(configs_dict['epochs']):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: configs_dict['dropout']})

            reg_loss = 0 
            if epoch/10 %2 == 0:
                _, training_loss, pred_error, attentive_reg, weight_reg = sess.run([model.model_weights_opt, model.loss,  model.pred_error, model.attentive_reg, model.weight_reg], feed_dict=feed_dict)
            else:
                _, training_loss, pred_error, attentive_reg, weight_reg = sess.run([model.label_prop_opt, model.loss,  model.pred_error, model.attentive_reg, model.weight_reg], feed_dict=feed_dict)

            # Validation
            loss_train, acc_train, duration = evaluate(sess, model, features, support, y_train, train_mask, placeholders)
            loss_val, acc_val, duration = evaluate(sess, model, features, support, y_val, val_mask, placeholders)
            loss_test, acc_test, duration = evaluate(sess, model, features, support, y_test, test_mask, placeholders)
            
            cost_val.append(loss_val)
            epoch_stats = [epoch+1, training_loss, pred_error, attentive_reg, weight_reg, loss_train, acc_train, loss_val, acc_val, loss_test, acc_test]
            training_log.append(epoch_stats)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), 
                "training_loss=", "{:.5f}".format(training_loss), "pred_error=", "{:.5f}".format(pred_error), 
                "attentive_reg=", "{:.5f}".format(attentive_reg),"weight_reg=", "{:.5f}".format(weight_reg), 
                "train_loss=", "{:.5f}".format(loss_train),"train_acc=", "{:.5f}".format(acc_train), 
                "val_loss=", "{:.5f}".format(loss_val), "val_acc=", "{:.5f}".format(acc_val), 
                "test_loss=", "{:.5f}".format(loss_test), "test_acc=", "{:.5f}".format(acc_test), 
                "time=", "{:.5f}".format(time.time() - t))

            if loss_val < best_config_val_loss:
                patience_count = 0
                best_config_val_loss = loss_val
                best_config_stats = epoch_stats
            else:
                patience_count += 1
            
            if patience_count > configs_dict['early_stopping']:
                print("Early stopping...")
                break

        print("Optimization Finished!")
        training_log = pd.DataFrame(np.array(training_log), columns = column_header)
        training_log.to_csv(training_logs_dir+'/'+params_str+'.csv', index=False)         
        params_tmp = []
        for x in params_str.split('\t'):
            try:
                params_tmp.append(float(x.split('_')[-1]))
            except:
                params_tmp.append(x.split('_')[-1])

        val_logs.append(params_tmp + best_config_stats)
        if best_config_val_loss < best_val_loss:
            best_val_loss = best_config_val_loss
            best_configs = configs_dict
            best_stats = best_config_stats
            attentive_weights = []
            if configs_dict['is_attentive']:
                for i in range(configs_dict['n_layers']):
                    attentive_weights.append(sess.run(model.layers[i].vars['attentive_weights_0']))                
                    pkl_dump(output_dir + '/layer_%d_attentive_weights.pkl'%i, attentive_weights[-1])
                    best_configs['layer_%d_attentive_weights'%i] = attentive_weights[-1]
            if configs_dict['propagate_labels']:
                    lb_prop_wts = sess.run(model.layers[-1].vars['label_propagation_weights'])
                    pkl_dump(output_dir + '/lb_prop_wts.pkl', lb_prop_wts)
            
            pkl_dump(output_dir+'/configs_dict.pkl', best_configs)
            
    if config_count %100 == 0:
        val_logs_df = pd.DataFrame(np.array(val_logs), columns = [" ".join(x.split('_')[:-1]) for x in params_str.split('\t')] + column_header)
        val_logs_df.to_csv(output_dir+'/val_metrics.csv',index=False)

val_logs_df = pd.DataFrame(np.array(val_logs), columns = [" ".join(x.split('_')[:-1]) for x in params_str.split('\t')] + column_header)
val_logs_df.to_csv(output_dir+'/val_metrics.csv',index=False)

print('Results')
print("Epoch:", '%04d\n' % best_stats[column_map['Epoch']], 
    "Train Loss =", "{:.5f}\n".format(best_stats[column_map['Train_Loss']]), "Train Accuracy =", "{:.5f}\n".format(best_stats[column_map['Train_Acc']]), 
    "Val Loss =", "{:.5f}\n".format(best_stats[column_map['Val_Loss']]), "Val Accuracy =", "{:.5f}\n".format(best_stats[column_map['Val_Acc']]), 
    "Test Loss =", "{:.5f}\n".format(best_stats[column_map['Test_Loss']]), "Test Accuracy =", "{:.5f}\n".format(best_stats[column_map['Test_Acc']]))
