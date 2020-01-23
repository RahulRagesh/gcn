from inits import *
import tensorflow as tf


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def reset_layer_uid():
    _LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights', collections=[tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,  'model_weights'])
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias', collections=[tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,  'model_weights'])

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, configs, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.configs = configs 
        self.is_attentive = configs.get('is_attentive',False)
        if self.is_attentive:
            self.indices = [support.indices for support in self.support]
            self.num_indices = configs['feat_prop_num_indices']
            self.num_nodes = configs['num_nodes']
            self.attentive_feat_prop_init = tf.constant(configs['attentive_feat_prop_init'], dtype=tf.float32)

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i), collections=[tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,  'model_weights'])
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias', collections=[tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,  'model_weights'])

            if self.is_attentive:
                for i in range(len(self.support)):
                    self.vars['attentive_weights_' + str(i)] = from_tensor(configs['attentive_feat_prop_init'],(self.num_indices,), name='attentive_weights_' + str(i), collections = [tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,  'attentive_adj'])
                    self.support[i] = tf.SparseTensor(self.indices[i], tf.identity(self.vars['attentive_weights_' + str(i)]), (self.num_nodes,self.num_nodes))
        
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class NonLinearFeatTransform(Layer):
    """Non Linear Feature Transform layer."""
    def __init__(self, configs, placeholders, input_dim, output_dim, sparse_inputs=False, logging =True, n_layers = -1, act=tf.nn.relu, dropout=True, **kwargs):
        super(NonLinearFeatTransform, self).__init__(**kwargs)
        
        self.act = act
        self.configs = configs 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.logging = logging
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        if self.logging:
            self._log_vars()
        
        self.n_layers = self.configs['n_feat_layers'] if n_layers == -1 else n_layers

    def _call(self, inputs):
        if self.n_layers == 1:
            layer = Dense(input_dim=self.input_dim,
                        output_dim=self.output_dim,
                        placeholders=self.placeholders,
                        act= lambda x : x,
                        dropout=self.dropout,
                        sparse_inputs=self.sparse_inputs,
                        logging=self.logging)
            x = layer(inputs)
            self.vars = {**self.vars, **layer.vars}

        else:
            layer =  Dense(input_dim=self.input_dim,
                        output_dim=self.configs['feat_transform_dim'],
                        placeholders=self.placeholders,
                        act=tf.nn.relu,
                        dropout=self.dropout,
                        sparse_inputs=self.sparse_inputs,
                        logging=self.logging)
            
            x = layer(inputs)
            self.vars = {**self.vars, **layer.vars}
                        
            for i in range(self.n_layers-2):
                layer = Dense(input_dim=self.configs['feat_transform_dim'],
                            output_dim=self.configs['feat_transform_dim'],
                            placeholders=self.placeholders,
                            act=tf.nn.relu,
                            dropout=self.dropout,
                            sparse_inputs=False,
                            logging=self.logging)
                x = layer(x)
                self.vars = {**self.vars, **layer.vars}


            layer = Dense(input_dim=self.configs['feat_transform_dim'],
                            output_dim=self.output_dim,
                            placeholders=self.placeholders,
                            act= lambda x : x,
                            dropout=self.dropout,
                            sparse_inputs=False,
                            logging=self.logging)
            x = layer(x)
            self.vars = {**self.vars, **layer.vars}

        return self.act(x)



class FeatureAggregation(Layer):
    """Feature Aggregation layer."""
    def __init__(self, configs, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False,
                 logging=True,
                 **kwargs):
        super(FeatureAggregation, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.configs = configs 
        self.logging = logging
        self.is_attentive = configs.get('is_attentive',False)
        if self.is_attentive:
            self.indices = [support.indices for support in self.support]
            self.num_indices = configs['feat_prop_num_indices']
            self.num_nodes = configs['num_nodes']
            self.attentive_feat_prop_init = tf.constant(configs['attentive_feat_prop_init'], dtype=tf.float32)

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        if self.is_attentive:
            with tf.variable_scope(self.name + '_vars'):
                for i in range(len(self.support)):
                    self.vars['attentive_weights_' + str(i)] = from_tensor(configs['attentive_feat_prop_init'],(self.num_indices,), name='attentive_weights_' + str(i), collections = [tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,  'attentive_adj'])
                    self.support[i] = tf.SparseTensor(self.indices[i], tf.identity(self.vars['attentive_weights_' + str(i)]), (self.num_nodes,self.num_nodes))
        
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        supports = list()
        for i in range(len(self.support)):
            support = dot(self.support[i], x, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        return self.act(output)


class LabelPropagation(Layer):
    """Label Propagation layer."""
    def __init__(self, configs, placeholders, act=tf.nn.relu, **kwargs):
        super(LabelPropagation, self).__init__(**kwargs)
        
        self.act = act
        self.configs = configs 
        self.learnable_label_propagation = configs.get('learnable_label_propagation',False)
        self.label_adj_matrix = tf.SparseTensor(*configs['label_adj_matrix']) 


        if self.learnable_label_propagation:
            self.learable_aggregator_matrix = tf.SparseTensor(*configs['label_aggregator_matrix'])
            self.indices = self.label_adj_matrix.indices
            self.num_indices = configs['label_prop_num_indices']
            self.num_nodes = configs['num_nodes']
            self.vars['label_propagation_weights'] = from_tensor(configs['attentive_label_prop_init'], (self.num_indices,1), name='label_weights', collections = [tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,  'attentive_label'])
            self.vars['label_propagation_weights'] = tf.exp(self.vars['label_propagation_weights'])
            self.norm_vector = dot(self.learable_aggregator_matrix, self.vars['label_propagation_weights'],sparse=True)
            self.vars['label_propagation_weights'] = tf.reshape(self.vars['label_propagation_weights'] / self.norm_vector, (-1,))
            
            self.adj = tf.SparseTensor(self.indices, self.vars['label_propagation_weights'], (self.num_nodes,self.num_nodes))
        else:
            self.adj = self.label_adj_matrix

            
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        output = dot(self.adj, inputs, sparse=True)
        return self.act(output)
