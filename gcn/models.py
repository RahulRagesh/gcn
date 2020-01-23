from layers import *
from metrics import *

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        if self.configs['is_attentive']:
            self.attentive_adj_opt = self.optimizer.minimize(self.loss, var_list = tf.get_collection('attentive_adj'))
        if self.configs['learnable_label_propagation'] and self.configs['propagate_labels']:
            self.label_prop_opt = self.optimizer.minimize(self.loss, var_list = tf.get_collection('attentive_label'))

        self.model_weights_opt = self.optimizer.minimize(self.loss, var_list = tf.get_collection('model_weights'))
        self.opt_op = self.optimizer.minimize(self.loss, var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, ckpt, sess):
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, ckpt)
        print("Model saved in file: %s" % save_path)

    def load(self, ckpt, sess):
        saver = tf.train.Saver(self.vars)
        saver.restore(sess, ckpt)
        print("Model restored from file: %s" % ckpt)


class MLP(Model):
    def __init__(self, placeholders, configs, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.configs = configs
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.configs['learning_rate'])

        self.build()

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for key, var in layer.vars.items():
                if not key.startswith('attentive'):
                    self.loss += self.configs['weight_decay'] * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=self.configs['hidden1'],
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=self.configs['hidden1'],
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, configs, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.configs = configs
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.configs['learning_rate'])
        self.is_attentive = configs.get('is_attentive',False)
        self.num_indices = configs.get('num_indices',0)
        self.num_nodes = configs.get('num_nodes',0)

        self.propagate_labels = configs.get('propagate_labels',False)

        self.build()

    def _loss(self):
        # Weight Regularization
        self.weight_reg = 0
        i = 0
        for layer in self.layers:
            if isinstance(layer,GraphConvolution):
                for key, var in layer.vars.items():
                    if not key.startswith('attentive'):
                        self.weight_reg += self.configs['weight_decay'][i] * tf.nn.l2_loss(var)
                i = i + 1
        self.loss+=self.weight_reg

        # Cross entropy error
        self.pred_error = masked_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask']) 
        self.loss += self.pred_error

        # Attentive Regularization
        self.attentive_reg = tf.constant(0.0)
        i = 0        
        if self.is_attentive:
            for layer in self.layers:
                if isinstance(layer,GraphConvolution):
                    for key, var in layer.vars.items():
                        if key.startswith('attentive'):
                            self.attentive_reg += self.configs['attentive_reg'][i] * tf.nn.l2_loss(var - layer.attentive_feat_prop_init)
                    i = i + 1           
            self.loss += self.attentive_reg

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        
        if self.configs['n_prop_layers'] == 1:
            self.layers.append(GraphConvolution(configs = self.configs,
                                            input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        else:
            self.layers.append(GraphConvolution(configs = self.configs,
                                            input_dim=self.input_dim,
                                            output_dim=self.configs['hidden1'],
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

            for i in range(self.configs['n_prop_layers']-2):
                self.layers.append(GraphConvolution(configs = self.configs,
                                                    input_dim=self.configs['hidden1'],
                                                    output_dim=self.configs['hidden1'],
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.relu,
                                                    dropout=True,
                                                    logging=self.logging))
    
            self.layers.append(GraphConvolution(configs = self.configs,
                                                input_dim=self.configs['hidden1'],
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))

        self.layers.append(tf.nn.softmax)
        
        if self.propagate_labels:
            self.layers.append(LabelPropagation(configs = self.configs,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                logging=self.logging))
        

    def predict(self):
        return self.outputs


class NGCN(Model):
    def __init__(self, placeholders, configs, input_dim, **kwargs):
        super(NGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.configs = configs
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.configs['learning_rate'])
        self.is_attentive = configs.get('is_attentive',False)
        self.num_indices = configs.get('num_indices',0)
        self.num_nodes = configs.get('num_nodes',0)

        self.propagate_labels = configs.get('propagate_labels',False)

        self.build()

    def _loss(self):
        # Weight Regularization
        self.weight_reg = tf.constant(0.0)
        i = 0
        for layer in self.layers:
            if isinstance(layer,GraphConvolution):
                for key, var in layer.vars.items():
                    if not key.startswith('attentive'):
                        self.weight_reg += self.configs['weight_decay'][i] * tf.nn.l2_loss(var)
                i = i + 1
        self.loss+=self.weight_reg

        # Cross entropy error
        self.pred_error = masked_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask']) 
        self.loss += self.pred_error

        # Attentive Regularization
        self.attentive_reg = tf.constant(0.0)
        i = 0        
        if self.is_attentive:
            for layer in self.layers:
                if isinstance(layer,GraphConvolution):
                    for key, var in layer.vars.items():
                        if key.startswith('attentive'):
                            self.attentive_reg += self.configs['attentive_reg'][i] * tf.nn.l2_loss(var - layer.attentive_feat_prop_init)
                    i = i + 1           
            self.loss += self.attentive_reg

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(NonLinearFeatTransform(configs = self.configs,
                                                    input_dim=self.input_dim,
                                                    output_dim=self.configs['feat_transform_dim'],
                                                    placeholders=self.placeholders,
                                                    dropout=True,
                                                    sparse_inputs=True,
                                                    logging=self.logging))


        if self.configs['n_prop_layers'] == 1:
            self.layers.append(GraphConvolution(configs = self.configs,
                                            input_dim=self.configs['feat_transform_dim'],
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        else:
            self.layers.append(GraphConvolution(configs = self.configs,
                                            input_dim=self.configs['feat_transform_dim'],
                                            output_dim=self.configs['hidden1'],
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

            for i in range(self.configs['n_prop_layers']-2):
                self.layers.append(GraphConvolution(configs = self.configs,
                                                    input_dim=self.configs['hidden1'],
                                                    output_dim=self.configs['hidden1'],
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.relu,
                                                    dropout=True,
                                                    logging=self.logging))
    
            self.layers.append(GraphConvolution(configs = self.configs,
                                                input_dim=self.configs['hidden1'],
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))

        self.layers.append(tf.nn.softmax)
        
        if self.propagate_labels:
            self.layers.append(LabelPropagation(configs = self.configs,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                logging=self.logging))
        

    def predict(self):
        return self.outputs


class FANLT(Model):
    def __init__(self, placeholders, configs, input_dim, **kwargs):
        super(FANLT, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.configs = configs
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.configs['learning_rate'])
        self.is_attentive = configs.get('is_attentive',False)
        self.num_indices = configs.get('num_indices',0)
        self.num_nodes = configs.get('num_nodes',0)

        self.propagate_labels = configs.get('propagate_labels',False)

        self.build()

    def _loss(self):
        # Weight Regularization
        self.weight_reg = tf.constant(0.0)
        i = 0
        for layer in self.layers:
            if isinstance(layer,FeatureAggregation):
                for key, var in layer.vars.items():
                    if not key.startswith('attentive'):
                        self.weight_reg += self.configs['weight_decay'][i] * tf.nn.l2_loss(var)
                i = i + 1
        self.loss+=self.weight_reg

        # Cross entropy error
        self.pred_error = masked_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask']) 
        self.loss += self.pred_error

        # Attentive Regularization
        self.attentive_reg = tf.constant(0.0)
        i = 0        
        if self.is_attentive:
            for layer in self.layers:
                if isinstance(layer,FeatureAggregation):
                    for key, var in layer.vars.items():
                        if key.startswith('attentive'):
                            self.attentive_reg += self.configs['attentive_reg'][i] * tf.nn.l2_loss(var - layer.attentive_feat_prop_init)
                    i = i + 1           
            self.loss += self.attentive_reg

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        
        for i in range(self.configs['n_prop_layers']):
            if self.configs['n_prop_layers'] == 1:
                input_dim = self.input_dim
                output_dim = self.output_dim
                sparse_inputs = True
            elif i == self.configs['n_prop_layers']-1:
                input_dim = self.configs['feat_transform_dim']
                output_dim = self.output_dim
                sparse_inputs = False
            elif i==0:
                sparse_inputs = True 
                input_dim = self.input_dim
                output_dim = self.configs['feat_transform_dim']
            else:
                input_dim = self.configs['feat_transform_dim']
                output_dim = self.configs['feat_transform_dim']
                sparse_inputs = False

            self.layers.append(NonLinearFeatTransform(configs = self.configs,
                                    input_dim=input_dim,
                                    output_dim=self.configs['feat_transform_dim'],
                                    placeholders=self.placeholders,
                                    dropout=True,
                                    sparse_inputs=sparse_inputs))

            self.layers.append(FeatureAggregation(configs = self.configs,
                                                    placeholders=self.placeholders,
                                                    dropout=True,
                                                    sparse_inputs=False,
                                                    logging=self.logging))
            
            self.layers.append(Dense(input_dim=self.configs['feat_transform_dim'],
                                    output_dim=output_dim,
                                    placeholders=self.placeholders,
                                    dropout=True,
                                    sparse_inputs=False))

        self.layers.append(tf.nn.softmax)
        
        if self.propagate_labels:
            self.layers.append(LabelPropagation(configs = self.configs,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                logging=self.logging))
        

    def predict(self):
        return self.outputs
