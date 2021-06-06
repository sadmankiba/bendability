import keras
import sys
import numpy as np

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, concatenate, ReLU, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf
from scipy.stats import spearmanr

from tensorflow.keras.layers import Lambda
from tensorflow import keras

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ConvolutionLayer(Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 data_format,
                 padding='valid',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 __name__ = 'ConvolutionLayer',
                 **kwargs):
        super(ConvolutionLayer, self).__init__(filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            **kwargs)
        self.run_value = 1

    def call(self, inputs):
        if self.run_value > 2:
            x_tf = self.kernel
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            alpha = 20
            beta = 1/alpha
            bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(lambda x:
                                  tf.math.scalar_mul(beta, tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(alpha, x),
                                  tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis = 1), axis = 1)),
                                  tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.subtract(tf.math.scalar_mul(alpha, x),
                                  tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis = 1), axis = 1))), axis = 1)), axis = 1)),
                                  tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]])))), x_tf)
            transf = tf.transpose(filt_list, [1, 2, 0])
            # tf.print(filt_list, output_stream=sys.stderr)
            outputs = self._convolution_op(inputs, transf) ## type of outputs is <class 'tensorflow.python.framework.ops.Tensor'>

        else:
            outputs = self._convolution_op(inputs, self.kernel)
        self.run_value += 1
        return outputs

class nn_model:
    def __init__(self, dim_num, filters, kernel_size, pool_type, regularizer, activation_type, epochs, batch_size, loss_func, optimizer):
        """initialize basic parameters"""
        self.dim_num = dim_num
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.regularizer = regularizer
        self.activation_type = activation_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.optimizer = optimizer

      
    def create_model(self):
        # different metric functions
        def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))
        def spearman_fn(y_true, y_pred):
            return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                   tf.cast(y_true, tf.float32)], Tout=tf.float32)

        # building model
        # To build this model with the functional API,
        # you would start by creating an input node:
        forward = keras.Input(shape=(self.dim_num[1],self.dim_num[2]), name = 'forward')
        reverse = keras.Input(shape=(self.dim_num[1],self.dim_num[2]), name = 'reverse')

        first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', input_shape=(self.dim_num[1],self.dim_num[2]), use_bias = True)
        # first_layer = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, strides=1, data_format='channels_last', use_bias = True)
        #fw = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, strides=1, data_format='channels_last', use_bias = True)(forward)
        #rc = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, strides=1, data_format='channels_last', use_bias = True)(reverse)

        fw = first_layer(forward)
        rc = first_layer(reverse)

        concat = concatenate([fw, rc], axis=1)
        #concat = Add()([fw, rc])
        pool_size_input = concat.shape[1]

        concat_relu = ReLU()(concat)

        if self.pool_type == 'Max':
            pool_layer = MaxPooling1D(pool_size=4, strides=2)(concat_relu)
            #pool_layer = MaxPooling1D(pool_size=12)(concat_relu)
        elif self.pool_type == 'Ave':
            pool_layer = AveragePooling1D(pool_size=pool_size_input)(concat_relu)
        elif self.pool_type == 'custom':
            def out_shape(input_shape):
                shape = list(input_shape)
                print(input_shape)
                shape[0] = 10
                return tuple(shape)
            #model.add(Lambda(top_k, arguments={'k': 10}))
            def top_k(inputs, k):
                # tf.nn.top_k Finds values and indices of the k largest entries for the last dimension
                print(inputs.shape)
                inputs2 = tf.transpose(inputs, [0,2,1])
                new_vals = tf.nn.top_k(inputs2, k=k, sorted=True).values
                # transform back to (None, 10, 512)
                return tf.transpose(new_vals, [0,2,1])

            pool_layer = Lambda(top_k, arguments={'k': 2})(concat_relu)
            pool_layer = AveragePooling1D(pool_size=2)(pool_layer)
        elif self.pool_type == 'custom_sum':
            ## apply relu function before custom_sum functions
            def summed_up(inputs):
                #nonzero_vals = tf.keras.backend.relu(inputs)
                new_vals = tf.math.reduce_sum(inputs, axis = 1, keepdims = True)
                return new_vals
            pool_layer = Lambda(summed_up)(concat_relu)

        else:
            raise NameError('Set the pooling layer name correctly')

        flat = Flatten()(pool_layer)
        # dropout = Dropout(0.1)(flat)

        if self.regularizer == 'L_1':
            outputs = Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= 'relu')(flat)
        elif self.regularizer == 'L_2':
            outputs = Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= 'relu')(flat)
        else:
            raise NameError('Set the regularizer name correctly')

        outputs2 = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= 'linear')(outputs)
        model = keras.Model(inputs=[forward, reverse], outputs=outputs2)

        model.summary()

        if self.loss_func == 'mse':
            model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'huber':
            loss_huber = keras.losses.Huber(delta=1)
            model.compile(loss=loss_huber, optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'mae':
            loss_mae = keras.losses.MeanAbsoluteError()
            model.compile(loss=loss_mae, optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'rank_mse':
            model.compile(loss=rank_mse, optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        elif self.loss_func == 'poisson':
            loss_poisson = tf.keras.losses.Poisson()
            model.compile(loss=loss_poisson, optimizer=self.optimizer, metrics = [coeff_determination, spearman_fn])
        else:
            raise NameError('Unrecognized Loss Function')

        return model
