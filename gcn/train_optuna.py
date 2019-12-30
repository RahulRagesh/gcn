from __future__ import division
from __future__ import print_function

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from itertools import product
from gcn.utils import *
from gcn.models import GCN, MLP
import sys 
import ast 
from shutil import copyfile
import pandas as pd 
import optuna 


class Objective(object):
    def __init__(self, config_file, verbose=False):
        self.verbose = verbose
        config_dict = load_config_file(config_file)

        configs = ast.literal_eval(config_dict['configs'])
        self.output_dir = create_dir(configs['output_dir'])
        self.training_params_dict = ast.literal_eval(config_dict['training_params'])
        self.feature_prop_params_dict = ast.literal_eval(config_dict['feature_prop_params'])

        self.label_prop_params_dict = ast.literal_eval(config_dict['label_prop_params'])
        self.ssl_params_dict = ast.literal_eval(config_dict['ssl_params'])
        self.optuna_params_dict = ast.literal_eval(config_dict['optuna_params'])
        self.configs_dict = {**configs, **self.training_params_dict, **self.feature_prop_params_dict, **self.label_prop_params_dict, **self.ssl_params_dict, **self.optuna_params_dict}

        copyfile(config_file, self.output_dir+'/config.txt')
        self.training_logs_dir = create_dir(self.output_dir+'/training_logs')

        self.configs_dict['output_dir'] = self.output_dir
        
        # Load data
        self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = load_data(configs['dataset'])

        # Add Self Nodes
        self.adj_feat = self.adj + sp.eye(self.adj.shape[0])
        for k in range(2,self.feature_prop_params_dict['feat_prop_k']):
            self.adj_feat = self.adj_feat @ self.adj_feat 
        self.adj_feat.data = np.ones_like(self.adj_feat.data)

        # Add Self Nodes
        self.adj_label = self.adj + sp.eye(self.adj.shape[0])
        for k in range(2,self.label_prop_params_dict['label_prop_k']):
            self.adj_label = self.adj_label @ self.adj_label 
        self.adj_label.data = np.ones_like(self.adj_label.data)

        # Some preprocessing
        self.features = preprocess_features(self.features)
        if configs['model'] == 'gcn':
            self.support = [preprocess_adj(self.adj_feat)]
            self.num_supports = 1
            self.model_func = GCN
        elif configs['model'] == 'gcn_cheby':
            self.support = chebyshev_polynomials(self.adj_feat, configs['max_degree'])
            self.num_supports = 1 + configs['max_degree']
            self.model_func = GCN
        elif configs['model'] == 'dense':
            self.support = [preprocess_adj(self.adj_feat)]  # Not used
            self.num_supports = 1
            self.model_func = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(configs['model']))

        self.configs_dict['attentive_feat_prop_init'] = np.reshape(self.support[0][1], (-1,))
        self.configs_dict['feat_prop_num_indices'] =  self.support[0][0].shape[0]


        self.configs_dict['label_adj_matrix'] = sparse_to_tuple(row_normalize_sparse_matrix(self.adj_label))
        self.configs_dict['attentive_label_prop_init'] = np.reshape(self.configs_dict['label_adj_matrix'][1], (-1,1))

        if self.configs_dict.get('propagate_labels',False):
            if self.configs_dict.get('learnable_label_propagation',False):
                self.configs_dict['label_aggregator_matrix'] = indices_to_aggregator(sparse_to_tuple(self.adj_label)[0])
            else:
                self.configs_dict['label_aggregator_matrix'] = self.configs_dict['label_adj_matrix']
        self.configs_dict['label_prop_num_indices'] = self.configs_dict['label_adj_matrix'][0].shape[0]

        self.configs_dict['num_nodes'] = self.support[0][2][0]
        self.column_header = ['Epoch', 'Training_Loss', 'Pred_Error', 'Attentive_Reg', 'Weight_Reg', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'Test_Loss', 'Test_Acc']

    def __call__(self, trial):
        tf.reset_default_graph()    

        seed = 123
        np.random.seed(seed)
        tf.set_random_seed(seed)
        wt_reg = []
        for i in range(self.configs_dict['n_layers']):
            if  self.training_params_dict['weight_decay'][0] == 'Sweep':
                if self.training_params_dict['weight_decay'][1] =='LogUniform':
                    wt_reg.append(trial.suggest_loguniform('Wt_Reg_%d'%i, self.training_params_dict['weight_decay'][2], self.training_params_dict['weight_decay'][3]))
                else:
                    wt_reg.append(trial.suggest_uniform('Wt_Reg_%d'%i, self.training_params_dict['weight_decay'][2], self.training_params_dict['weight_decay'][3]))
                self.configs_dict['weight_decay'] = tuple(wt_reg)
            else:
                self.configs_dict['weight_decay'] = self.training_params_dict['weight_decay'][1]

        if  self.training_params_dict['learning_rate'][0] == 'Sweep':
            if self.training_params_dict['learning_rate'][1] =='LogUniform':
                    self.configs_dict['learning_rate'] = trial.suggest_loguniform('learning_rate', self.training_params_dict['learning_rate'][2], self.training_params_dict['learning_rate'][3])
            else:
                    self.configs_dict['learning_rate'] = trial.suggest_uniform('learning_rate', self.training_params_dict['learning_rate'][2], self.training_params_dict['learning_rate'][3])
        else:
            self.configs_dict['learning_rate'] =  self.training_params_dict['learning_rate'][1]

        if  self.training_params_dict['dropout'][0] == 'Sweep':
            if self.training_params_dict['dropout'][1] =='LogUniform':
                self.configs_dict['dropout'] = trial.suggest_loguniform('dropout', self.training_params_dict['dropout'][2], self.training_params_dict['dropout'][3])
            else:
                self.configs_dict['dropout'] = trial.suggest_uniform('dropout', self.training_params_dict['dropout'][2], self.training_params_dict['dropout'][3])
        else:
            self.configs_dict['dropout'] =  self.training_params_dict['dropout'][1]
        
        if self.configs_dict['is_attentive']:
            attentive_reg = []
            for i in range(self.configs_dict['n_layers']):
                if  self.feature_prop_params_dict['attentive_reg'][0] == 'Sweep':
                    if self.feature_prop_params_dict['attentive_reg'][1] =='LogUniform':
                        attentive_reg.append(trial.suggest_loguniform('Attentive_Reg_%d'%i, self.feature_prop_params_dict['attentive_reg'][2], self.feature_prop_params_dict['attentive_reg'][3]))
                    else:
                        attentive_reg.append(trial.suggest_uniform('Attentive_Reg_%d'%i, self.feature_prop_params_dict['attentive_reg'][2], self.feature_prop_params_dict['attentive_reg'][3]))                
                    self.configs_dict['attentive_reg'] = tuple(attentive_reg)
                else:
                    self.configs_dict['attentive_reg'] = self.training_params_dict['attentive_reg'][1]
                
   
        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(self.num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }
        
        # Create model
        model = self.model_func(placeholders, self.configs_dict, input_dim=self.features[2][1], logging=True)

        with tf.Session() as sess:                
            sess.run(tf.global_variables_initializer())
            # Train model
            training_log = []
            best_config_acc = 0
            best_config_loss = 1000000
            for epoch in range(self.configs_dict['epochs']):

                t = time.time()
                # Construct feed dictionary
                feed_dict = construct_feed_dict(self.features, self.support, self.y_train, self.train_mask, placeholders)
                feed_dict.update({placeholders['dropout']: self.configs_dict['dropout']})
                reg_loss = 0 
                '''
                if epoch//10%2==0:
                    _, training_loss, pred_error, attentive_reg, weight_reg = sess.run([model.attentive_adj_opt, model.loss,  model.pred_error, model.attentive_reg, model.weight_reg], feed_dict=feed_dict)
                else:   
                    _, training_loss, pred_error, attentive_reg, weight_reg = sess.run([model.model_weights_opt, model.loss,  model.pred_error, model.attentive_reg, model.weight_reg], feed_dict=feed_dict)
                '''
                _, training_loss, pred_error, attentive_reg, weight_reg = sess.run([model.opt_op, model.loss,  model.pred_error, model.attentive_reg, model.weight_reg], feed_dict=feed_dict)

                # Validation
                loss_train, acc_train, duration = evaluate(sess, model, self.features, self.support, self.y_train, self.train_mask, placeholders)
                loss_val, acc_val, duration = evaluate(sess, model, self.features, self.support, self.y_val, self.val_mask, placeholders)
                loss_test, acc_test, duration = evaluate(sess, model, self.features, self.support, self.y_test, self.test_mask, placeholders)
                
                epoch_stats = [epoch+1, training_loss, pred_error, attentive_reg, weight_reg, loss_train, acc_train, loss_val, acc_val, loss_test, acc_test]
                training_log.append(epoch_stats)

                if self.verbose:
                    # Print results
                    print("Epoch:", '%04d' % (epoch + 1), 
                        "training_loss=", "{:.5f}".format(training_loss), "pred_error=", "{:.5f}".format(pred_error), 
                        "attentive_reg=", "{:.5f}".format(attentive_reg),"weight_reg=", "{:.5f}".format(weight_reg), 
                        "train_loss=", "{:.5f}".format(loss_train),"train_acc=", "{:.5f}".format(acc_train), 
                        "val_loss=", "{:.5f}".format(loss_val), "val_acc=", "{:.5f}".format(acc_val), 
                        "test_loss=", "{:.5f}".format(loss_test), "test_acc=", "{:.5f}".format(acc_test), 
                        "time=", "{:.5f}".format(time.time() - t))
                
                if self.configs_dict['direction'] == 'minimize':
                    if loss_val <= best_config_loss:
                        patience_count = 0
                        best_config_loss = loss_val
                        trial.set_user_attr('Epoch', epoch+1)
                        trial.set_user_attr('Train Accuracy', acc_train)
                        trial.set_user_attr('Train Loss', loss_train)
                        trial.set_user_attr('Val Accuracy', acc_val)
                        trial.set_user_attr('Val Loss', loss_val)
                        trial.set_user_attr('Test Accuracy', acc_test)
                        trial.set_user_attr('Test Loss', loss_test)
                        result = best_config_loss
                    else:
                        patience_count += 1
                else:
                    if acc_val >= best_config_acc:
                        patience_count = 0
                        best_config_acc = acc_val
                        trial.set_user_attr('Epoch', epoch+1)
                        trial.set_user_attr('Train Accuracy', acc_train)
                        trial.set_user_attr('Train Loss', loss_train)
                        trial.set_user_attr('Val Accuracy', acc_val)
                        trial.set_user_attr('Val Loss', loss_val)
                        trial.set_user_attr('Test Accuracy', acc_test)
                        trial.set_user_attr('Test Loss', loss_test)
                        result = best_config_acc
                    else:
                        patience_count += 1

                if patience_count > self.configs_dict['early_stopping']:
                    print("Early stopping...")
                    training_log = pd.DataFrame(np.array(training_log), columns = self.column_header)
                    training_log.to_csv(self.training_logs_dir+'/'+'trial_%05d.csv'%trial.number, index=False)
                    
                    self.config_dir = self.output_dir + '/trial_%05d/'%trial.number
                    os.makedirs(self.config_dir)
                    
                    attentive_weights = []
                    if self.configs_dict['is_attentive']:
                        for i in range(self.configs_dict['n_layers']):
                            attentive_weights.append(sess.run(model.layers[i].vars['attentive_weights_0']))                
                            pkl_dump(self.config_dir + '/layer_%d_attentive_weights.pkl'%i, attentive_weights[-1])
                        
                    if self.configs_dict['propagate_labels'] and self.configs_dict['learnable_label_propagation']:
                        lb_prop_wts = sess.run(model.layers[-1].vars['label_propagation_weights'])
                        pkl_dump(self.config_dir + '/lb_prop_wts.pkl', lb_prop_wts)
                    
                    return best_config_loss  
            return best_config_loss  

def trial_log_callback(study, trial):
    print('\nTRIAL ', trial.number)
    print('Train Accuracy - ', trial.user_attrs['Train Accuracy'])
    print('Val Accuracy - ', trial.user_attrs['Val Accuracy'])
    print('Test Accuracy - ', trial.user_attrs['Test Accuracy'])
    print()
    if trial.number%100 == 0:
        print('\n-----------------------------------------------------------')
        print('-----------------------------------------------------------')
        best_trial = study.best_trial
        print('BEST TRIAL N0 - ', best_trial.number)
        print('Train Accuracy - ', best_trial.user_attrs['Train Accuracy'])
        print('Val Accuracy - ', best_trial.user_attrs['Val Accuracy'])
        print('Test Accuracy - ', best_trial.user_attrs['Test Accuracy'])
        print()
        for key, value in trial.params.items():
            print("%s - %f"%(key,value))
        print('-----------------------------------------------------------')
        print('-----------------------------------------------------------\n')


config_dict = load_config_file(sys.argv[1])
optuna_params_dict = ast.literal_eval(config_dict['optuna_params'])
study = optuna.create_study(direction=optuna_params_dict['direction'])
optuna.logging.disable_default_handler()
objective = Objective(sys.argv[1], verbose = optuna_params_dict['verbose'])
study.optimize(objective, n_trials=optuna_params_dict['n_trials'], callbacks = [trial_log_callback])
trial = study.best_trial 

print("Best Hyperparameters")
for key, value in trial.params.items():
    print(key,' : ',value)

pkl_dump(objective.output_dir+'/best_config.pkl',trial.params)


print("\n\nBest Metrics")
print('BEST TRIAL N0    - ', trial.number)
print('Train Accuracy   - ', trial.user_attrs['Train Accuracy'])
print('Train Loss       - ', trial.user_attrs['Train Loss'])
print('Val Accuracy     - ', trial.user_attrs['Val Accuracy'])
print('Val Loss         - ', trial.user_attrs['Val Loss'])
print('Test Accuracy    - ', trial.user_attrs['Test Accuracy'])
print('Test Loss        - ', trial.user_attrs['Test Loss'])

pkl_dump(objective.output_dir+'/best_metrics.pkl',trial.user_attrs)

df = study.trials_dataframe()
df.to_csv(objective.output_dir+'/config_metrics.csv',index=False)
