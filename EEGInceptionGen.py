# Built-in imports
import math

# External imports
import tensorflow as tf
from tensorflow.keras.layers import Activation, Input, concatenate
from tensorflow.keras.layers import Dropout, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import Conv2D, AveragePooling2D, DepthwiseConv2D
from tensorflow.keras.layers import Dense, SpatialDropout2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras as keras
import sklearn.utils as sk_utils
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class EEGInceptionGen:
    """EEGInceptionGen is a general model for EEG classification tasks. It is
    the second version of EEG-Inception [1].

    References
    ----------
    [1] Santamaría-Vázquez, E., Martínez-Cagigal, V., Vaquerizo-Villar, F., &
    Hornero, R. (2020). EEG-Inception: A Novel Deep Convolutional Neural Network
    for Assistive ERP-based Brain-Computer Interfaces. IEEE Transactions on
    Neural Systems and Rehabilitation Engineering.
    """
    def __init__(self, input_time=1000, fs=128, n_cha=8, n_inception_blocks=1,
                 inception_blocks_filters_per_branch=8,
                 scales_time=(500, 250, 125), n_spatial_filt_mult=2,
                 output_pooling_factor=2,
                 dropout_type='Dropout', dropout_rate=0.25,
                 activation='elu', n_classes=2, learning_rate=0.001):
        """
        Class constructor

        Parameters
        ----------
        input_time: int
            Length of the input epochs in time (ms)
        fs: float
            Sample rate of the input signal
        n_cha: int
            Number of EEG chanels
        n_inception_blocks:
            Number of Inception blocks before and after the spatial layer
        inception_blocks_filters_per_branch: int
            Number of filters in each Inception branch
        scales_time: tuple
            Size of the receptive field of temporal convolutions.
        n_spatial_filt_mult: int
            Multiplier of the depthwise convolution layer along the spatial axis
        output_pooling_factor: int
            Pooling factor in each layer of the output module
        dropout_type: str ['Dropout', 'SpatialDropout2D']
            Dropout type. SpatialDropout has more regularization power, but it
            does not as good as Dropout.
        dropout_rate: float
            Dropout rate in all dropout layers
        activation:
            Activation function after all convolutions
        n_classes:
            Number of classes
        learning_rate:
            Learning rate for training
        """
        # Dropout
        dropout_type_str = dropout_type
        if dropout_type_str == 'Dropout':
            dropout_type = Dropout
        elif dropout_type_str == 'SpatialDropout2D':
            dropout_type = SpatialDropout2D
        else:
            raise ValueError('Dropout_type must be one of %s' %
                             str(['Dropout', 'SpatialDropout2D']))

        # Super call
        super().__init__(fit=[], predict_proba=['y_pred'])

        # Tensorflow config
        tf.keras.backend.set_image_data_format('channels_last')

        # Parameters
        self.input_time = input_time
        self.fs = fs
        self.n_cha = n_cha
        self.n_inception_blocks = n_inception_blocks
        self.inception_blocks_filters_per_branch = \
            inception_blocks_filters_per_branch
        self.scales_time = scales_time
        self.n_spatial_filt_mult = n_spatial_filt_mult
        self.output_pooling_factor = output_pooling_factor
        self.dropout_type_str = dropout_type_str
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.n_classes = n_classes
        self.learning_rate = learning_rate

        # Useful variables
        self.input_samples = int(input_time * fs / 1000)
        self.scales_samples = [int(s * fs / 1000) for s in scales_time]

        # Create model
        self.model = self.__keras_model()
        # Create training callbacks
        self.training_callbacks = list()
        self.training_callbacks.append(
            EarlyStopping(monitor='val_loss',
                          min_delta=0.001,
                          mode='min',
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)
        )
        # Create fine-tuning callbacks
        self.fine_tuning_callbacks = list()
        self.fine_tuning_callbacks.append(
            EarlyStopping(monitor='val_loss',
                          min_delta=0.0001,
                          mode='min',
                          patience=5,
                          verbose=1,
                          restore_best_weights=True)
        )

    def one_hot_labels(self, categorical_labels):
        """One hot encoding labels
        """
        enc = OneHotEncoder(handle_unknown='ignore')
        one_hot_labels = \
            enc.fit_transform(categorical_labels.reshape(-1, 1)).toarray()

        return one_hot_labels

    def get_scales(self, pooling_track):
        """Returns the scales taking into account the average pooling applied
        in earlier layers
        """
        pooling_factor = np.prod(pooling_track)
        scales = [int(sc // pooling_factor) for sc in self.scales_samples]
        return scales

    def __basic_temp_conv2d_block(self, input_layer, n_filt, kernel_size):
        """Basic conv block of EEG-InceptionGen"""
        if not isinstance(kernel_size, tuple) and \
                not isinstance(kernel_size, list):
            raise ValueError('Parameter kernel_size must be tuple or list')
        u = Conv2D(filters=n_filt,
                   kernel_size=kernel_size,
                   kernel_initializer='he_normal',
                   padding='same')(input_layer)
        u = BatchNormalization()(u)
        u = Activation(self.activation)(u)
        return u

    def __inception_module(self, input_layer, scales_samples):
        """Creates an Inception module
        """
        u2 = list()
        for i in range(len(scales_samples)):
            # Temporal conv
            branch = self.__basic_temp_conv2d_block(
                input_layer, self.inception_blocks_filters_per_branch, (scales_samples[i], 1))
            u2.append(branch)

        b1_out = concatenate(u2, axis=3)
        return b1_out

    def __keras_model(self):
        """ Builds the keras model
         """
        # Number of branches
        n_branches = len(self.scales_samples)
        # Number of output filters from the inception blocks
        n_out_filt_inception = \
            n_branches * self.inception_blocks_filters_per_branch
        # Pooling tracking
        pooling_track = list()

        # ============================= INPUT ================================ #
        input_layer = Input((self.input_samples, self.n_cha, 1))

        # ================ BLOCK 1: TEMPORAL INCEPTION MODULES =============== #
        # Inception blocks
        scales = self.get_scales(pooling_track)
        b1 = self.__inception_module(input_layer, scales)
        for i in range(self.n_inception_blocks-1):
            b1 = self.__inception_module(b1, scales)

        # Average pooling
        b1 = AveragePooling2D((2, 1))(b1)
        pooling_track.append(2)

        # Dropout
        b1 = self.dropout_type(self.dropout_rate)(b1)

        # ==================== BLOCK 2: CHANNEL SELECTION ==================== #
        b2 = DepthwiseConv2D((1, self.n_cha),
                             depth_multiplier=self.n_spatial_filt_mult,
                             padding='valid',
                             depthwise_constraint=max_norm(1.))(b1)
        b2 = BatchNormalization()(b2)
        b2 = Activation(self.activation)(b2)

        # Dropout
        b2 = self.dropout_type(self.dropout_rate)(b2)

        # ============= BLOCK 3: SPATIO-TEMPORAL INCEPTION MODULES =========== #
        scales = self.get_scales(pooling_track)
        b3 = self.__inception_module(b2, scales)
        for i in range(self.n_inception_blocks - 1):
            b3 = self.__inception_module(b3, scales)

        # Average pooling
        b3 = AveragePooling2D((2, 1))(b3)
        pooling_track.append(2)

        # Dropout
        b3 = self.dropout_type(self.dropout_rate)(b3)

        # ==================== BLOCK 4: OUTPUT-BLOCK ========================= #
        # Params
        input_samples_to_out_block = b3.shape[1]
        n_out_blocks = math.log(input_samples_to_out_block,
                                self.output_pooling_factor)
        n_out_blocks = math.ceil(n_out_blocks) - 1
        n_out_filt_increase_ratio = 4

        # Output blocks
        filt_increase = int(np.ceil(input_samples_to_out_block /
                                    self.output_pooling_factor /
                                    n_out_filt_increase_ratio))
        filt_increase = max(filt_increase,
                            n_out_filt_increase_ratio)
        n_out_filt_inception += filt_increase
        max_scale = max(self.get_scales(pooling_track))
        max_scale = 1 if max_scale <= 0 else max_scale
        b4 = self.__basic_temp_conv2d_block(input_layer=b3,
                                            n_filt=n_out_filt_inception,
                                            kernel_size=(max_scale, 1))
        b4 = AveragePooling2D((self.output_pooling_factor, 1))(b4)
        pooling_track.append(self.output_pooling_factor)
        for i in range(n_out_blocks - 1):
            filt_increase = int(np.ceil(filt_increase /
                                        self.output_pooling_factor))
            filt_increase = max(filt_increase,
                                n_out_filt_increase_ratio)
            n_out_filt_inception += filt_increase
            max_scale = max(self.get_scales(pooling_track))
            max_scale = 1 if max_scale <= 0 else max_scale
            b4 = self.__basic_temp_conv2d_block(input_layer=b4,
                                                n_filt=n_out_filt_inception,
                                                kernel_size=(max_scale, 1))
            b4 = AveragePooling2D((self.output_pooling_factor, 1))(b4)
            pooling_track.append(self.output_pooling_factor)

        # =========================== OUTPUT ================================= #
        # Global average pooling + dropout
        b4 = GlobalAvgPool2D()(b4)
        b4 = Dropout(self.dropout_rate)(b4)

        # Output layer
        output_layer = Dense(self.n_classes, activation='softmax')(b4)

        # ============================ MODEL ================================= #
        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate,
                                          beta_1=0.9, beta_2=0.999,
                                          amsgrad=False)
        # Create and compile model
        model = keras.models.Model(inputs=input_layer,
                                   outputs=output_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def __transform_data(self, X, y=None):
        """Transforms input data to the correct dimensions for EEG-Inception

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        y: np.ndarray
            Labels array. If labels are in categorical format, they will be
            converted to one-hot encoding.
        """
        if len(X.shape) == 3 or X.shape[-1] != 1:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        if y is None:
            return X
        else:
            if len(y.shape) == 1 or y.shape[-1] == 1:
                y = self.one_hot_labels(y)
            return X, y

    def fit(self, X, y, fine_tuning=False, shuffle_before_fit=False, **kwargs):
        """Fit the model. Additional keras parameters of class
        tensorflow.keras.Model will pass through. See keras documentation to
        know what can you do: https://keras.io/api/models/model_training_apis/.

        If no parameters are specified, some default options are set [1]:

            - Epochs: 100 if fine_tuning else 500
            - Batch size: 32 if fine_tuning else 1024
            - Callbacks: self.fine_tuning_callbacks if fine_tuning else
                self.training_callbacks

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        y: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        fine_tuning: bool
            Set to True to use the default training parameters for fine
            tuning. False by default.
        shuffle_before_fit: bool
            If True, the data will be shuffled before training just once. Note
            that if you use the keras native argument 'shuffle', the data is
            shuffled each epoch.
        kwargs:
            Key-value arguments will be passed to the fit function of the model.
            This way, you can set your own training parameters using keras API.
            See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        """
        # Shuffle the data before fitting
        if shuffle_before_fit:
            X, y = sk_utils.shuffle(X, y)

        # Training parameters
        if not fine_tuning:
            # Rewrite default values
            kwargs['epochs'] = kwargs['epochs'] if \
                'epochs' in kwargs else 500
            kwargs['batch_size'] = kwargs['batch_size'] if \
                'batch_size' in kwargs else 2048
            kwargs['callbacks'] = kwargs['callbacks'] if \
                'callbacks' in kwargs else self.training_callbacks
        else:
            kwargs['epochs'] = kwargs['epochs'] if \
                'epochs' in kwargs else 100
            kwargs['batch_size'] = kwargs['batch_size'] if \
                'batch_size' in kwargs else 32
            kwargs['callbacks'] = kwargs['callbacks'] if \
                'callbacks' in kwargs else self.fine_tuning_callbacks

        # Transform data if necessary
        X, y = self.__transform_data(X, y)
        # Fit
        return self.model.fit(X, y, **kwargs)

    def predict_proba(self, X):
        """Model prediction scores for the given features.

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels]
        """
        # Transform data if necessary
        X = self.__transform_data(X)
        # Predict
        return self.model.predict(X)



