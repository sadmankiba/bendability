import keras

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, concatenate, ReLU, Maximum, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf
from scipy.stats import spearmanr

from tensorflow.keras.layers import Lambda
from tensorflow import keras

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class ConvolutionLayer(Conv1D):
    def __init__(self, alpha, beta,
                 filters,
                 kernel_size,
                 data_format,
                 padding='valid',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 __name__='ConvolutionLayer',
                 **kwargs):
        super(ConvolutionLayer, self).__init__(filters=filters,
                                               kernel_size=kernel_size,
                                               activation=activation,
                                               use_bias=use_bias,
                                               kernel_initializer=kernel_initializer,
                                               data_format=data_format,
                                               padding=padding,
                                               **kwargs)
        self.run_value = 1
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        if self.run_value > 2:

            x_tf = self.kernel  # x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(lambda x:
                                  tf.math.scalar_mul(self.beta, tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(self.alpha, x),
                                                                                                    tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis=1), axis=1)),
                                                                                        tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.subtract(tf.math.scalar_mul(self.alpha, x),
                                                                                                                                                              tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis=1), axis=1))), axis=1)), axis=1)),
                                                                            tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]])))), x_tf)
            transf = tf.transpose(filt_list, [1, 2, 0])
            outputs = self._convolution_op(inputs, transf)

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
            SS_res = K.sum(K.square(y_true-y_pred))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))

        def spearman_fn(y_true, y_pred):
            return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                                              tf.cast(y_true, tf.float32)], Tout=tf.float32)
        
        def coeff_determination_loss(y_true, y_pred):
            return 1 - coeff_determination(y_true, y_pred)

        # building model
        # To build this model with the functional API,
        # you would start by creating an input node:
        forward = keras.Input(
            shape=(self.dim_num[1], self.dim_num[2]), name='forward')
        reverse = keras.Input(
            shape=(self.dim_num[1], self.dim_num[2]), name='reverse')

        first_layer_1 = ConvolutionLayer(
            alpha=75.0, beta=1/75.0, filters=256, kernel_size=8, strides=1, data_format='channels_last', use_bias=True, padding='same')

        fw_1 = first_layer_1(forward)
        rc_1 = first_layer_1(reverse)

        relu_fw_1 = ReLU()(fw_1)
        relu_rc_1 = ReLU()(rc_1)
        relu_rv_1 = tf.reverse(relu_fw_1, axis=[1])
        relu_co_1 = tf.reverse(relu_rc_1, axis=[1])

        arra_1 = concatenate([relu_fw_1, relu_co_1])
        arra_2 = concatenate([relu_rc_1, relu_rv_1])

        conv_24 = Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            dilation_rate=8,
            data_format='channels_last',
            use_bias=True,
            kernel_initializer='normal',
            kernel_regularizer=regularizers.l2(0.0005), # 0.0005 for data 9, 0.05 for other data
            padding='same',
        )

        arra_1_24 = conv_24(arra_1)
        arra_2_24 = conv_24(arra_2)

        relu_arra_1_24 = ReLU()(arra_1_24)
        relu_arra_2_24 = ReLU()(arra_2_24)

        max_24 = Maximum()([relu_arra_1_24, relu_arra_2_24])

        output_conv = Conv1D(
            filters=1,
            kernel_size=max_24.shape[1],
            data_format='channels_last',
            use_bias=True,
            kernel_initializer='normal',
            kernel_regularizer=regularizers.l2(0.0005), # 0.0005 for data 9, 0.05 for other data
            activation='linear'
        )

        outputs = Flatten()(output_conv(max_24))

        model = keras.Model(inputs=[forward, reverse], outputs=outputs)

        model.summary()

        adam_optimizer = keras.optimizers.Adam()

        if self.loss_func == 'mse':
            model.compile(loss='mean_squared_error', optimizer=adam_optimizer, metrics=[
                          coeff_determination, spearman_fn])
        elif self.loss_func == 'coeff_determination':
            model.compile(loss=coeff_determination_loss, optimizer=adam_optimizer, metrics=[
                          keras.metrics.mean_squared_error, coeff_determination, spearman_fn])
        elif self.loss_func == 'huber':
            loss_huber = keras.losses.Huber(delta=1)
            model.compile(loss=loss_huber, optimizer=self.optimizer,
                          metrics=[coeff_determination, spearman_fn])
        elif self.loss_func == 'mae':
            loss_mae = keras.losses.MeanAbsoluteError()
            model.compile(loss=loss_mae, optimizer=self.optimizer,
                          metrics=[coeff_determination, spearman_fn])
        elif self.loss_func == 'rank_mse':
            model.compile(loss='rank_mse', optimizer=self.optimizer,
                          metrics=[coeff_determination, spearman_fn])
        elif self.loss_func == 'poisson':
            loss_poisson = tf.keras.losses.Poisson()
            model.compile(loss=loss_poisson, optimizer=self.optimizer, metrics=[
                          coeff_determination, spearman_fn])
        else:
            raise NameError('Unrecognized Loss Function')

        return model
