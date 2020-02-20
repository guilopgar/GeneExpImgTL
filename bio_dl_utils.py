## Module containing auxiliary functions and classes for completing methodology implementation

# Discrete fixed-time survival
def survival_fixed_time(fix_time, time, event):
    """
    It returns 0 if the individual survives the fixed time point.
    Else, pit returns 1 if the event occurs before the fixed time point.
    None is returned if the individual is censored before the fixed time point.
    """
    if (time > fix_time): return 0
    else:
        if (event == 1.0): return 1
        else: return None


# Image shape transformer
def reshape_transformer(X, final_shape):
    return X.reshape(X.shape[0], *final_shape)


# Performance metrics
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, recall_score, accuracy_score, precision_score, f1_score, matthews_corrcoef

def optimal_threshold(y_true, y_prob):
    # y_prob: Probability of class 1. Shape [n_samples]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    opt_i = np.argmax(tpr - fpr)
    # Return optimal threshold
    return thresholds[opt_i]

def optimal_conf_matrix(y_true, y_prob):
    # Probability of class 1
    #y_prob = y_prob[:, 0]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    opt_i = np.argmax(tpr - fpr)
    c = confusion_matrix(y_true, (y_prob > thresholds[opt_i])*1)
    # Return confusion matrix computed using the optimal threshold
    return c

def opt_sensitivity_score(y_true, y_prob):
    c = optimal_conf_matrix(y_true, y_prob)
    return c[1][1]/(c[1][1] + c[1][0])

def opt_specificity_score(y_true, y_prob):
    c = optimal_conf_matrix(y_true, y_prob)
    return c[0][0]/(c[0][0] + c[0][1])

def opt_recall_score(y_true, y_prob):
    # Probability of class 1
    #y_prob = y_prob[:, 0]
    opt_t = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob > opt_t)*1
    return recall_score(y_true, y_pred)

def opt_accuracy_score(y_true, y_prob):
    # Probability of class 1
    #y_prob = y_prob[:, 0]
    opt_t = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob > opt_t)*1
    return accuracy_score(y_true, y_pred)

def opt_precision_score(y_true, y_prob):
    # Probability of class 1
    #y_prob = y_prob[:, 0]
    opt_t = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob > opt_t)*1
    return precision_score(y_true, y_pred)

def opt_f1_score(y_true, y_prob):
    # Probability of class 1
    #y_prob = y_prob[:, 0]
    opt_t = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob > opt_t)*1
    return f1_score(y_true, y_pred)

def opt_mcc_score(y_true, y_prob):
    # Probability of class 1
    #y_prob = y_prob[:, 0]
    opt_t = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob > opt_t)*1
    return matthews_corrcoef(y_true, y_pred)

def opt_threshold_score(y_true, y_prob):
    # Probability of class 1
    #y_prob = y_prob[:, 0]
    # Return optimal threshold
    return optimal_threshold(y_true, y_prob)


# Classes and functions used to create and estimate DL models
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from keras.layers import Input, Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, RMSprop, Adam

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import roc_auc_score

from multiprocessing import Process, Pool

from hyperopt import fmin, tpe, STATUS_OK, Trials
from functools import partial

class EarlyRocAUC(Callback):
    """
    Custom callback that performs early-stopping monitoring Roc-AUC on validation data.
    
    Arguments:
        val_x: Numpy array containing the validation dataset.
        val_y: Numpy array containing the validation fixed time point survival data.
        patience: Patience epochs used during early-stopping training.
    """
    
    def __init__(self, val_x, val_y, patience=5):
        super(Callback, self).__init__()
        self.X_val = val_x
        self.y_val = val_y
        self.patience = patience
        

    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None


    def on_epoch_end(self, epoch, logs=None):
        current_roc_auc = roc_auc_score(y_true=self.y_val, y_score=self.model.predict(self.X_val, batch_size=150, verbose=0))
        
        if (current_roc_auc > self.best):
            self.best = current_roc_auc
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)



def create_dense_layers(input_layer, dense_unit=None, dense_activation=None, dense_dropout=None, output_unit=None, 
                       output_activation=None):
    """
    It creates a stack of densely-connected layers and an output layer 
    on top of a given layer.

    """

    # Create args default values
    if dense_unit is None: dense_unit = []
    if dense_activation is None: dense_activation = []
    if dense_dropout is None: dense_dropout = []
    if output_unit is None: output_unit = 1
    if output_activation is None: output_activation = 'sigmoid'
    
    # Check dense layers args
    if not (len(dense_unit) == len(dense_activation) == len(dense_dropout)):
        raise ValueError("'dense_unit', 'dense_activation' and 'dense_dropout' " + 
                         "arguments must have the same length")

    # Create the stack of hidden layers
    layer = input_layer
    for dense_u, dense_a, dense_d in zip(dense_unit, dense_activation, dense_dropout):
        layer = Dense(dense_u)(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(dense_a)(layer)
        layer = Dropout(dense_d)(layer)
            
    # Create the output layer
    output_layer = Dense(units=output_unit, activation=output_activation)(layer)
    
    return output_layer



def create_mlnn(input_shape, dense_unit=None, dense_activation=None, dense_dropout=None, output_unit=None, 
               output_activation=None):
    """
    Generic function that creates a Keras MLNN model, which consists of a series of dense layers (dense,
    batch-normalization, activation and dropout), and an output layer.
    
    :param input_shape: Tuple of integers containing the input shape of the model. The last dimension is assumed 
                        to contain the number of channels.
    :param cnn_filter: List of integers containing the number of filters used in each conv2d layer.
    :param cnn_kernel: List of integers containing the size of the squared kernel used in each conv2d layer.
    :param cnn_activation: List of strings containing the name of the activation function used in each conv2d layer.
    :param cnn_pool: List of integers containing the size of the squared max-pool downscale factor used in each conv2d layer.
    :param cnn_dropout: List of real numbers containing the dropout rate used in each conv2d layer.
    :param dense_unit: List of integers containing the number of units used in each dense layer.
    :param dense_activation: List of strings containing the name of the activation function used in each dense layer.
    :param dense_dropout: List of real numbers containing the dropout rate used in each dense layer.
    :param output_unit: Integer as the number of units used in the output layer.
    :param output_activation: String as the name of the activation function used the output layer.
    
    :returns: Keras functional model, which is not compiled.
    """
    
    # Create the input layer
    input_layer = Input(shape=(input_shape,))
    
    output_layer = create_dense_layers(input_layer=input_layer, dense_unit=dense_unit, dense_activation=dense_activation, 
        dense_dropout=dense_dropout, output_unit=output_unit, output_activation=output_activation)
    
    return Model(input_layer, output_layer)



def create_cnn(input_shape, cnn_filter, cnn_kernel, cnn_activation, cnn_pool, cnn_dropout, 
               dense_unit=None, dense_activation=None, dense_dropout=None, output_unit=None, 
               output_activation=None):
    """
    Generic function that creates a Keras 2D-CNN model, which consists of a series of conv2d layers (convolution filter,
    batch-normalization, activation, max-pooling and dropout), a series of dense layers (dense, batch-normalization, 
    activation and dropout) and an output layer.
    
    It is specially designed to use squared matrices as input data.
    
    :param input_shape: Tuple of integers containing the input shape of the model. The last dimension is assumed 
                        to contain the number of channels.
    :param cnn_filter: List of integers containing the number of filters used in each conv2d layer.
    :param cnn_kernel: List of integers containing the size of the squared kernel used in each conv2d layer.
    :param cnn_activation: List of strings containing the name of the activation function used in each conv2d layer.
    :param cnn_pool: List of integers containing the size of the squared max-pool downscale factor used in each conv2d layer.
    :param cnn_dropout: List of real numbers containing the dropout rate used in each conv2d layer.
    :param dense_unit: List of integers containing the number of units used in each dense layer.
    :param dense_activation: List of strings containing the name of the activation function used in each dense layer.
    :param dense_dropout: List of real numbers containing the dropout rate used in each dense layer.
    :param output_unit: Integer as the number of units used in the output layer.
    :param output_activation: String as the name of the activation function used the output layer.
    
    :returns: Keras functional model, which is not compiled.
    """
    
    # Check conv2d layers args
    if not (len(cnn_filter) == len(cnn_kernel) == len(cnn_activation) == len(cnn_pool) == len(cnn_dropout)):
        raise ValueError("'cnn_filter', 'cnn_kernel', 'cnn_activation', 'cnn_pool', and 'cnn_dropout' " + 
                         "arguments must have the same length")
    
    # Create the input layer
    input_layer = Input(shape=input_shape)
    
    # Create the conv2d layers
    layer = input_layer
    for cnn_f, cnn_k, cnn_a, cnn_p, cnn_d in zip(cnn_filter, cnn_kernel, cnn_activation, cnn_pool, cnn_dropout):
        # int used because of skopt
        layer = Conv2D(filters=cnn_f, kernel_size=cnn_k, data_format="channels_last")(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(cnn_a)(layer)
        layer = MaxPool2D(pool_size=cnn_p, data_format="channels_last")(layer)
        layer = Dropout(cnn_d)(layer)
    
    # Create flaten layer
    layer = Flatten(data_format="channels_last")(layer)
    
    # Add a stack of densely-connected layers
    output_layer = create_dense_layers(input_layer=layer, dense_unit=dense_unit, dense_activation=dense_activation, 
        dense_dropout=dense_dropout, output_unit=output_unit, output_activation=output_activation)
    
    return Model(input_layer, output_layer)



# HERE GUILLE NOW, ERROR: When predicting, take into account that, if output unit is 1, keras model.predict 
# (equivalent to sklear model.predict_proba) will return an array with shape [n_samples], not [n_samples, n_labels], 
# as output is only one (sigmoid activation function)

class SklearnMLNN(BaseEstimator, ClassifierMixin):
    """
    Abstract generic Keras model that acts as a sklearn classifier. It is designed to overcome the tensorflow
    GPU memory release bug (https://github.com/tensorflow/tensorflow/issues/17048). Before executing, 
    no tensorflow session can be active. 
    
    Default arguments values are set to perform discrete-time survival prediction.
    
    
    Parameters
    ----------
    model_path : string, default 'keras-models/keras_ann.h5'
        It stands for the file path of the Keras swallow sequential model that is saved and loaded
        during training and prediction procedures.
    
    n_input : integer, default 100
        Number of units used at the input layer.
    
    n_hidden : integer, default 70
        Number of units used at the hidden layer.
    
    n_output : integer, default 39
        Number of units used at the output layer.
    
    batch_norm : bool, default True
        If True, Batch-Normalization layer is used at the hidden layer before applying activation function.
    
    f_hidden : string, default 'relu'
        Keras activation function used by the hidden layer.
    
    f_output : string, default 'sigmoid'
        Keras activation function used by the output layer.
        
    f_loss : function or string, default SurvivalPickable(39)
        The loss function used to train the model. If string, it must be the name of an 
        available Keras loss function.
    
    opt_name: string, default 'sgd'
        Name of the Keras optimizer used to train the model. 
        Available names are 'sgd', 'rmsprop' and 'adam'.
        
    lr: numeric, default 0.0005
        Learning rate used by the optimizer during training.
    
    momentum: numeric, default 0.5
        Momentum used by the SGD algorithm during training.
        
    nesterov: bool, default True
        If True, SGD algorithm uses Nesterov momentum during training.
    
    n_epoch: integer, default 20
        Number of epochs to train the model.
    
    batch_size: integer, default 40
        Number of samples per gradient update used during training.
    
    verbose : integer, default 0
        Verbose parameter of the 'fit' keras function.
    
    output_unit: if 1, the model is assumed to be trained on a binary classifcation problem.


    Attributes
    ----------
    _X : array-like, shape [n_samples, n_features]
         Input dataset.

    _y : array-like, shape [n_samples]
         Class labels used to perform model training.
    
    _optimizer : Keras optimizer object
         Keras optimizer used to perform model training.
    """
    
    def __init__(self, input_shape, dense_unit=None, dense_activation=None, dense_dropout=None, output_unit=1, 
                 output_activation=None, optimizer_name='adam', lr=0.001, momentum=0.5, nesterov=True, 
                 loss_function='binary_crossentropy', batch_size=32, epoch=10, patience=0, verbose=0, 
                 model_path='keras-models/mlnn_pre_train_pfi_hyperopt.h5'):

        self.input_shape = input_shape
        self.dense_unit = dense_unit
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        self.output_unit = output_unit
        self.output_activation = output_activation
        self.optimizer_name = optimizer_name
        self.lr = lr    
        self.momentum = momentum
        self.nesterov = nesterov
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience
        self.verbose = verbose
        self.model_path = model_path


    def _update(self, **kwargs):
        """
        Private auxiliary function to update dropout and activity regularizer values.
        
        Parameters
        ----------
        **kwargs : dict, {argument: value}
            'dropout_A' and 'activity_B' arguments are expected, where the name of the arguments represent the 
            layer (A or B) where the dropout or activity regularization is applied, and the values are either
            the dropout rates or the regularizer objects. Theses values are included in the dropout_dict and
            activity_dict instance attributes, respectively.
        """
        
        for key, value in kwargs.items():
            if 'dense_unit' in key:
                # Specific for Hyperopt returning float instead of int
                self.dense_unit[int(key.split('_')[-1])] = int(value)
            elif 'dense_activation' in key:
                self.dense_activation[int(key.split('_')[-1])] = value
            elif 'dense_dropout' in key:
                self.dense_dropout[int(key.split('_')[-1])] = value
        
        return self
    
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        """
        
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        non_valid_params = {}
        for key, value in params.items():
            if key not in valid_params:
                non_valid_params[key] = value
            else:
                setattr(self, key, value)

        self._update(**non_valid_params)

        return self


    def _create_model(self):
        """
        Private auxiliary function to create a MLNN Keras model.

        """
        
        return create_mlnn(input_shape=self.input_shape, output_unit=self.output_unit, output_activation=self.output_activation,
                         dense_unit=list(dict(sorted(self.dense_unit.items(), key=lambda x: x[0])).values()), 
                         dense_activation=list(dict(sorted(self.dense_activation.items(), key=lambda x: x[0])).values()), 
                         dense_dropout=list(dict(sorted(self.dense_dropout.items(), key=lambda x: x[0])).values()))
    
    
    def _train(self):
        """
        Private auxiliary function to perform model training, in order to fix the tensorflow GPU 
        memory release bug (https://github.com/tensorflow/tensorflow/issues/17048).

        """
        
        # Prevent GPU memory allocation problems
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        # Create Keras model
        clf = self._create_model()
        
        # Create Keras optimizer object
        if self.optimizer_name == 'sgd': self._optimizer = SGD(lr=self.lr, momentum=self.momentum, nesterov=self.nesterov)
        elif self.optimizer_name == 'rmsprop': self._optimizer = RMSprop(lr=self.lr)
        elif self.optimizer_name == 'adam': self._optimizer = Adam(lr=self.lr)
        else: raise ValueError("Invalid optimizer. Availble optimizers are: 'sgd', 'rmsprop' and 'adam'")
        
        # Compile the model
        clf.compile(optimizer=self._optimizer, loss=self.loss_function)
        
        if self.patience > 0:
            # Use 90% of data to train and 10% for early-stopping validation
            # Stratify according to the censoring time distribution across intervals
            split = list(StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=60).split(self._X, self._y))
            X_train = self._X[split[0][0], :]
            # Specific for 1-D label array
            y_train = self._y[split[0][0]]
            X_val = self._X[split[0][1], :]
            # Specific for 1-D label array
            y_val = self._y[split[0][1]]
            
            clf.fit(X_train, y_train, batch_size=int(self.batch_size), epochs=int(self.epoch), verbose=self.verbose,  
                    callbacks=[EarlyRocAUC(val_x=X_val, val_y=y_val, patience=self.patience)])
        
        # Specific for Hyperopt returning float instead of int
        else: clf.fit(self._X, self._y, batch_size=int(self.batch_size), epochs=int(self.epoch), verbose=self.verbose)
        
        # Save the model
        Model(clf.input, clf.output).save(self.model_path)
        
        return
        
    
    def fit(self, X, y):
        """
        Train the model using supervised learning.
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to train the model.
        
        y : array-like, shape [n_samples]
            The class label used to perform the supervised training. 
        """
        
        self._X = X
        self._y = y
        
        p = Process(target=self._train)
        p.start()
        p.join()
        
        return self
    
    
    def _clf_predict_proba(self):
        """
        Private uxiliary function to predict probabilities using the trained model, in order to fix the 
        tensorflow GPU memory release bug (https://github.com/tensorflow/tensorflow/issues/17048).
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data.
        
        Returns
        -------
        output : array-like, shape [n_samples, n_labels]
        """
        
        # Prevent GPU memory allocation problems
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        clf_fit = load_model(self.model_path)
        
        return clf_fit.predict(self._X, batch_size=150, verbose=0)
    
    
    def predict(self, X):
        """
        Predict labels with trained model.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data.

        Returns
        -------
        output : array-like, shape [n_samples]
        """
        
        # Dataset too big to transfer as arg to apply function
        self._X = X
        
        with Pool(1) as p:
            y_pred = p.apply(self._clf_predict_proba)
        
        return np.argmax(y_pred, axis=-1)
    
    
    def predict_proba(self, X):
        """
        Predict labels probaiblities with fitted estimator model.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data.

        Returns
        -------
        output : array-like, shape [n_samples, n_labels]
        """
        
        # Dataset too big to transfer as arg to apply function
        self._X = X
        
        with Pool(1) as p:
            y_pred_proba = p.apply(self._clf_predict_proba)
        
        # Binary classification
        if self.output_unit == 1:
            return np.concatenate((1 - y_pred_proba, y_pred_proba), axis=1)

        else: return y_pred_proba
        


# HERE GUILLE NOW, ERROR: When predicting, take into account that, if output unit is 1, keras model.predict 
# (equivalent to sklear model.predict_proba) will return an array with shape [n_samples], not [n_samples, n_labels], 
# as output is only one (sigmoid activation function)

class SklearnCNN(SklearnMLNN):
    """
    Custom sklearn classifier that acts as a Keras CNN model. It extends the simple MLNN architecture implemented 
    in SklearnMLNN class. Before executing, no tensorflow session can be active. 
    
    Default arguments values are set to perform discrete-time survival prediction.
    
    
    Parameters
    ----------
    model_path : string, default 'keras-models/keras_ann.h5'
        It stands for the file path of the Keras swallow sequential model that is saved and loaded
        during training and prediction procedures.
    
    n_input : integer, default 100
        Number of units used at the input layer.
    
    n_hidden : integer, default 70
        Number of units used at the hidden layer.
    
    n_output : integer, default 39
        Number of units used at the output layer.
    
    batch_norm : bool, default True
        If True, Batch-Normalization layer is used at the hidden layer before applying activation function.
    
    f_hidden : string, default 'relu'
        Keras activation function used by the hidden layer.
    
    f_output : string, default 'sigmoid'
        Keras activation function used by the output layer.
        
    f_loss : function or string, default SurvivalPickable(39)
        The loss function used to train the model. If string, it must be the name of an 
        available Keras loss function.
    
    opt_name: string, default 'sgd'
        Name of the Keras optimizer used to train the model. 
        Available names are 'sgd', 'rmsprop' and 'adam'.
        
    lr: numeric, default 0.0005
        Learning rate used by the optimizer during training.
    
    momentum: numeric, default 0.5
        Momentum used by the SGD algorithm during training.
        
    nesterov: bool, default True
        If True, SGD algorithm uses Nesterov momentum during training.
    
    n_epoch: integer, default 20
        Number of epochs to train the model.
    
    batch_size: integer, default 40
        Number of samples per gradient update used during training.
    
    verbose : integer, default 0
        Verbose parameter of the 'fit' keras function.


    Attributes
    ----------
    _X : array-like, shape [n_samples, n_features]
         Input dataset.

    _y : array-like, shape [n_samples]
         Class labels used to perform model training.
    
    _optimizer : Keras optimizer object
         Keras optimizer used to perform model training.
    """
    
    def __init__(self, input_shape, cnn_filter, cnn_kernel, cnn_activation, cnn_pool, cnn_dropout, dense_unit, 
                 dense_activation, dense_dropout, output_unit, output_activation=None, 
                 optimizer_name='adam', lr=0.001, momentum=0.5, nesterov=True, loss_function='binary_crossentropy', 
                 batch_size=32, epoch=10, patience=0, verbose=0, 
                 model_path='keras-models/cnn_pre_train_pfi_hyperopt.h5'):

        # Call constructor from parent class
        super(SklearnCNN, self).__init__(input_shape=input_shape, dense_unit=dense_unit, dense_activation=dense_activation, 
            dense_dropout=dense_dropout, output_unit=output_unit, output_activation=output_activation, 
            optimizer_name=optimizer_name, lr=lr, momentum=momentum, nesterov=nesterov, 
            loss_function=loss_function, batch_size=batch_size, epoch=epoch, patience=patience, 
            verbose=verbose, model_path=model_path)

        self.cnn_filter = cnn_filter
        self.cnn_kernel = cnn_kernel
        self.cnn_activation = cnn_activation
        self.cnn_pool = cnn_pool
        self.cnn_dropout = cnn_dropout


    def _update(self, **kwargs):
        """
        Overrides parent class method.

        Private auxiliary function to update dropout and activity regularizer values. 
        
        Parameters
        ----------
        **kwargs : dict, {argument: value}
            'dropout_A' and 'activity_B' arguments are expected, where the name of the arguments represent the 
            layer (A or B) where the dropout or activity regularization is applied, and the values are either
            the dropout rates or the regularizer objects. Theses values are included in the dropout_dict and
            activity_dict instance attributes, respectively.
        """

        # Call _update method from parent class
        super(SklearnCNN, self)._update(**kwargs)
        
        for key, value in kwargs.items():
            if 'cnn_filter' in key:
                # Specific for Hyperopt returning float instead of int
                self.cnn_filter[int(key.split('_')[-1])] = int(value)
            elif 'cnn_kernel' in key:
                # Specific for Hyperopt returning float instead of int
                self.cnn_kernel[int(key.split('_')[-1])] = int(value)
            elif 'cnn_activation' in key:
                self.cnn_activation[int(key.split('_')[-1])] = value
            elif 'cnn_pool' in key:
                # Specific for Hyperopt returning float instead of int
                self.cnn_pool[int(key.split('_')[-1])] = int(value)
            elif 'cnn_dropout' in key:
                self.cnn_dropout[int(key.split('_')[-1])] = value
        
        return self
    
    
    def _create_model(self):
        """
        Overrides parent class method.

        Private auxiliary function to create a 2D-CNN Keras model.

        """
        
        return create_cnn(input_shape=self.input_shape, output_unit=self.output_unit, 
                         output_activation=self.output_activation,
                         cnn_filter=list(dict(sorted(self.cnn_filter.items(), key=lambda x: x[0])).values()), 
                         cnn_kernel=list(dict(sorted(self.cnn_kernel.items(), key=lambda x: x[0])).values()), 
                         cnn_activation=list(dict(sorted(self.cnn_activation.items(), key=lambda x: x[0])).values()), 
                         cnn_pool=list(dict(sorted(self.cnn_pool.items(), key=lambda x: x[0])).values()), 
                         cnn_dropout=list(dict(sorted(self.cnn_dropout.items(), key=lambda x: x[0])).values()),
                         dense_unit=list(dict(sorted(self.dense_unit.items(), key=lambda x: x[0])).values()), 
                         dense_activation=list(dict(sorted(self.dense_activation.items(), key=lambda x: x[0])).values()), 
                         dense_dropout=list(dict(sorted(self.dense_dropout.items(), key=lambda x: x[0])).values()))



def create_fine_tune(pre_layer, n_freeze, pre_model=None, dense_unit=None, dense_activation=None, dense_dropout=None, 
                     output_unit=None, output_activation=None):
    """
    Generic function that creates a Sequential Keras 2D-CNN model using an already pre-trained 2D-CNN model. The new model is
    intended to be used for fine-tuning the network, and can be created in two different ways, either fine-tuning
    the previous model, or substituting the last layers by new final dense layers (including the output).
    
    :param pre_layer: Integer representing the number of layers from the pre-trained model to be included in the new model.
                      This number is expected to be either the number of total layers or the flatten layer index.
                      In the first case, the new model is obtained by fine-tuning the pre-trained model, 
                      and in the latter case the new model is created by adding new dense layers after the flatten layer
                      of the pre-trained model.
    :param n_freeze: Integer representing the number of layers to be freezed, i.e. the first n_freeze layers will
                     be set to be non-trainable.
    :param pre_model: String containing the file path of a saved Keras model.
    :param dense_unit: List of integers containing the number of units used in each new dense layer.
                       Only used if the model is obtained by adding new dense layers.
    :param dense_activation: List of strings containing the name of the activation function used in each new dense layer.
                             Only used if the model is obtained by adding new dense layers.
    :param dense_dropout: List of real numbers containing the dropout rate used in each new dense layer.
                          Only used if the model is obtained by adding new dense layers.
    :param output_unit: Integer as the number of units used in the output layer.
                        Only used if the model is obtained by adding new dense layers.
    :param output_activation: String as the name of the activation function used by the output layer.
                              Only used if the model is obtained by adding new dense layers.
    
    :returns: Keras Sequential model, which is not compiled.
    """
    
    # Create args default values
    if dense_unit is None: dense_unit = []
    if dense_activation is None: dense_activation = []
    if dense_dropout is None: dense_dropout = []
    if output_unit is None: output_unit = 1
    if output_activation is None: output_activation = 'sigmoid'
    if pre_model is None: pre_model = 'keras-models/hyperopt_es_cont_cnn_pre_train.h5',
    
    # Load pre-trained model
    pre_model = load_model(pre_model)
    
    # Create fine-tuning model
    fine_clf = Sequential()
    
    for i in range(pre_layer):
        # Freeze layer
        if i < n_freeze: pre_model.layers[i].trainable = False
        # Add layer
        fine_clf.add(pre_model.layers[i])
    
    if pre_layer < len(pre_model.layers):
        # Check dense layers args
        if not (len(dense_unit) == len(dense_activation) == len(dense_dropout)):
            raise ValueError("'dense_unit', 'dense_activation' and 'dense_dropout' " + 
                             "arguments must have the same length")
        # Add dense layers
        i = pre_layer
        for dense_u, dense_a, dense_d in zip(dense_unit, dense_activation, dense_dropout):
            fine_clf.add(Dense(dense_u, name="dense-" + str(i) + "-FT"))
            fine_clf.add(BatchNormalization(name="bn-" + str(i) + "-FT"))
            fine_clf.add(Activation(dense_a, name="act-" + str(i) + "-FT"))
            fine_clf.add(Dropout(dense_d, name="drop-" + str(i) + "-FT"))
            i += 1
        # Add output layer
        fine_clf.add(Dense(units=output_unit, activation=output_activation, name="output-FT"))
    
    return fine_clf



# HERE GUILLE NOW, ERROR: When predicting, take into account that, if output unit is 1, keras model.predict 
# (equivalent to sklear model.predict_proba) will return an array with shape [n_samples], not [n_samples, n_labels], 
# as output is only one (sigmoid activation function)

class SklearnFT(SklearnMLNN):
    """
    Custom sklearn classifier that acts as a Keras MLP model. It is designed to overcome the tensorflow
    GPU memory release bug (https://github.com/tensorflow/tensorflow/issues/17048). Before executing, 
    no tensorflow session can be active. 
    
    Default arguments values are set to perform discrete-time survival prediction.
    
    
    Parameters
    ----------
    model_path : string, default 'keras-models/keras_ann.h5'
        It stands for the file path of the Keras swallow sequential model that is saved and loaded
        during training and prediction procedures.
    
    n_input : integer, default 100
        Number of units used at the input layer.
    
    n_hidden : integer, default 70
        Number of units used at the hidden layer.
    
    n_output : integer, default 39
        Number of units used at the output layer.
    
    batch_norm : bool, default True
        If True, Batch-Normalization layer is used at the hidden layer before applying activation function.
    
    f_hidden : string, default 'relu'
        Keras activation function used by the hidden layer.
    
    f_output : string, default 'sigmoid'
        Keras activation function used by the output layer.
        
    f_loss : function or string, default SurvivalPickable(39)
        The loss function used to train the model. If string, it must be the name of an 
        available Keras loss function.
    
    opt_name: string, default 'sgd'
        Name of the Keras optimizer used to train the model. 
        Available names are 'sgd', 'rmsprop' and 'adam'.
        
    lr: numeric, default 0.0005
        Learning rate used by the optimizer during training.
    
    momentum: numeric, default 0.5
        Momentum used by the SGD algorithm during training.
        
    nesterov: bool, default True
        If True, SGD algorithm uses Nesterov momentum during training.
    
    n_epoch: integer, default 20
        Number of epochs to train the model.
    
    batch_size: integer, default 40
        Number of samples per gradient update used during training.
    
    verbose : integer, default 0
        Verbose parameter of the 'fit' keras function.

    len_dense: 4, as 4 Keras layers form a densely connected layer in the pre-trained model.

    add_dense: strongly related to pre_layer, they should be carefully jointly defined.

    pre_layer: it should point to the last layer of the pre-trained model before the
        stack of densely connected layers.
    
    output_unit: if 1, the model is assumed to be trained on a binary classifcation problem.




    Attributes
    ----------
    _X : array-like, shape [n_samples, n_features]
         Input dataset.

    _y : array-like, shape [n_samples]
         Class labels used to perform model training.
    
    _optimizer : Keras optimizer object
         Keras optimizer used to perform model training.
    """
    
    def __init__(self, pre_layer, n_freeze, pre_model=None, dense_unit=None, dense_activation=None, 
                 dense_dropout=None, output_unit=1, output_activation=None, optimizer_name='adam', 
                 lr=0.001, momentum=0.5, nesterov=True, loss_function='binary_crossentropy', batch_size=32, 
                 epoch=10, patience=0, verbose=0, model_path='keras-models/mlnn_pre_train.h5'):

        # Call constructor from parent class
        # input_shape has no sense here, as the model is loaded, not created from scratch
        super(SklearnFT, self).__init__(input_shape=None, dense_unit=dense_unit, dense_activation=dense_activation, 
            dense_dropout=dense_dropout, output_unit=output_unit, output_activation=output_activation, 
            optimizer_name=optimizer_name, lr=lr, momentum=momentum, nesterov=nesterov, 
            loss_function=loss_function, batch_size=batch_size, epoch=epoch, patience=patience, 
            verbose=verbose, model_path=model_path)

        self.pre_layer = pre_layer
        self.n_freeze = n_freeze
        self.pre_model = pre_model
    
    
    def _create_model(self):
        """
        Overrides parent class method.
        
        Private auxiliary function to create a Keras Sequential model from a pre-trained model.
        
        """
        
        return create_fine_tune(pre_layer=self.pre_layer, n_freeze=self.n_freeze, pre_model=self.pre_model,
                         output_unit=self.output_unit, output_activation=self.output_activation,
                         dense_unit=list(dict(sorted(self.dense_unit.items(), key=lambda x: x[0])).values()), 
                         dense_activation=list(dict(sorted(self.dense_activation.items(), key=lambda x: x[0])).values()), 
                         dense_dropout=list(dict(sorted(self.dense_dropout.items(), key=lambda x: x[0])).values()))



class HyperoptCV(BaseEstimator, ClassifierMixin):
    """
    Custom sklearn classifier that performs hyperopt-bayesian model selection of a certain estimator using 
    cross-validation, and then fits the model using the best estimated hyper-parameters configuration.
    
    Based on: 
    https://stackoverflow.com/questions/52408949/cross-validation-and-parameters-tuning-with-xgboost-and-hyperopt


    Parameters
    ----------
    estimator : sklearn estimator object
        This is assumed to implement the scikit-learn estimator interface.
    
    hyper_space : dict
        Dictionary whose keys are the name of hyper-parameters from the estimator to be optimized and 
        corresponding values are the hyperopt distributions of the hyper-parameters possible values.
    
    cv : sklearn cross-validation generator
        Cross-validation procedure used to perform model selection, average test scoring metric is considered.
        
    scoring : string, default 'accuracy'
        Scoring metric used during model selection.
    
    n_iter : int, default 20
        Number of iterations used during bayesian optimization, i.e. maximum number of evaluations performed by 
        the bayesian optimizer.
    
    random_seed : int, default None
        Integer seed used by the pseudo-random number generator during bayesian optimization.
    
    
    Attributes
    ----------
    best_estimator_: fitted sklearn estimator object
        Fitted estimator using the best estimated hyper-parameters configuration.
    
    best_trial_: dict
        Trial data describing the best estimated model.
    """
    
    def __init__(self, estimator, hyper_space, cv, scoring='accuracy', opt_metric='accuracy', train_score=False, n_iter=20, 
                 random_seed=None, parallel=1):
        self.estimator = estimator
        self.hyper_space = hyper_space
        self.cv = cv
        self.scoring = scoring
        self.opt_metric = opt_metric
        self.train_score = train_score
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.parallel = parallel
    
    
    def _flatten_nested_dict(self, v, k=None):
        """
        Recursive function to flatten a nested dict. As hyperopt allows to define nested parameters
        in the hyper-parameters space, the nested dict must be flattened before feeding it to the estimator.
        """
        
        return {rec_k: rec_v for v_k, v_v in v.items() for rec_k, rec_v in self._flatten_nested_dict(v_v, v_k).items()} \
            if isinstance(v, dict) else {k: v}
    
    
    def _objective(self, params, X, y):
        """
        Private auxiliary method that defines the function to be minimized by hyperopt.
        
        """
        
        self.estimator.set_params(**self._flatten_nested_dict(params))
        res_cv = cross_validate(self.estimator, X, y, scoring=self.scoring, cv=self.cv, 
                                return_train_score=self.train_score, n_jobs=self.parallel)
        return {'loss': 1. - np.mean(res_cv['test_' + self.opt_metric]), 'params': params, 'score': res_cv, 
                'status': STATUS_OK}
    
    
    def model_selection(self, X, y):
        """
        Perform cross-validation bayesian optimization using hyperopt.
        
        Parameters
        ----------
        X: array-like, shape [n_samples, n_features]
            Data.

        y : array-like, shape [n_samples]
            Labels.
            
        Returns
        -------
        output : Trial.best_trial
            The trial given the lowest loss value.
        """
    
        trials = Trials()
        # Use partial, as fn callable can only take one arg
        fmin(partial(self._objective, X=X, y=y), self.hyper_space, algo=tpe.suggest, trials=trials, max_evals=self.n_iter, 
             rstate=np.random.RandomState(self.random_seed))
        
        # Latter used to explore hyper-params influence in model performance
        self.trials_ = trials
        
        return trials.best_trial
        

    def fit(self, X, y):
        """
        Perform bayesian model selection and then fit the estimator using the best estimated hyper-parameters 
        configuration.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data.

        y : array-like, shape [n_samples]
            Labels.
        """

        # Estimate best hyper-params using X, y
        self.best_trial_ = self.model_selection(X, y)
        
        # Instantiate estimator model with optimized hyper-params and then fit it
        self.estimator = self.estimator.set_params(**self.best_trial_['result']['params'])
        self.best_estimator_ = self.estimator.fit(X, y)
        
        return self

    
    def predict(self, X):
        """
        Predict labels with fitted estimator model.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data.

        Returns
        -------
        output : array-like, shape [n_samples]
        """
        
        if not hasattr(self, 'best_estimator_'):
            raise NotFittedError('Call `fit` before `predict`.')
        else:
            return self.best_estimator_.predict(X)

        
    def predict_proba(self, X):
        """
        Predict labels probaiblities with fitted estimator model.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data.

        Returns
        -------
        output : array-like, shape [n_samples]
        """
        
        if not hasattr(self, 'best_estimator_'):
            raise NotFittedError('Call `fit` before `predict_proba`.')
        else:
            return self.best_estimator_.predict_proba(X)



class HyperoptCV_TL(HyperoptCV):
    """
    Custom sklearn classifier that performs hyperopt-bayesian model selection of a certain estimator using 
    cross-validation, and then fits the model using the best estimated hyper-parameters configuration.
    
    Based on: 
    https://stackoverflow.com/questions/52408949/cross-validation-and-parameters-tuning-with-xgboost-and-hyperopt


    Parameters
    ----------
    estimator : sklearn estimator object
        This is assumed to implement the scikit-learn estimator interface.
    
    hyper_space : dict
        Dictionary whose keys are the name of hyper-parameters from the estimator to be optimized and 
        corresponding values are the hyperopt distributions of the hyper-parameters possible values.
    
    cv : sklearn cross-validation generator
        Cross-validation procedure used to perform model selection, average test scoring metric is considered.
        
    scoring : string, default 'accuracy'
        Scoring metric used during model selection.
    
    n_iter : int, default 20
        Number of iterations used during bayesian optimization, i.e. maximum number of evaluations performed by 
        the bayesian optimizer.
    
    random_seed : int, default None
        Integer seed used by the pseudo-random number generator during bayesian optimization.
    
    
    Attributes
    ----------
    best_estimator_: fitted sklearn estimator object
        Fitted estimator using the best estimated hyper-parameters configuration.
    
    best_trial_: dict
        Trial data describing the best estimated model.
    """
    
    def __init__(self, estimator_pt, X_pt, y_pt, estimator_ft, hyper_space, cv, scoring='accuracy', opt_metric='accuracy', 
                 train_score=False, n_iter=20, random_seed=None, len_dense=4, verbose_file="verbose.txt"):
        
        super(HyperoptCV_TL, self).__init__(estimator=estimator_ft, hyper_space=hyper_space, cv=cv, scoring=scoring, 
            opt_metric=opt_metric, train_score=train_score, n_iter=n_iter, random_seed=random_seed, parallel=None)
        self.estimator_pt = estimator_pt
        self.X_pt = X_pt
        self.y_pt = y_pt
        self.len_dense = len_dense
        self.verbose_file = verbose_file
    
    
    def _objective(self, params, X, y):
        """
        Overrides parent class method.
        
        """
        
        # PT model
        ## Extract PT estimator hyper-parameters
        pt_params = self._flatten_nested_dict(params['clf__pt_params'])
        add_dense = pt_params.pop('clf__add_dense')
        self._estimator_pt = clone(self.estimator_pt, safe=True)
        self._estimator_pt.set_params(**pt_params)
        ## Print and Write verbose
        print("PT Hyper-params:")
        print(pt_params)
        print("PT params:")
        print(self._estimator_pt.get_params()['clf'])
        with open(self.verbose_file, 'a') as f:
            f.write("\nPT Hyper-params:\n")
            f.write(str(pt_params))
            f.write("\nPT params:\n")
            f.write(str(self._estimator_pt.get_params()['clf']))
        ## Fit model
        self._estimator_pt.fit(self.X_pt, self.y_pt)
        
        # FT model
        ## Extract FT estimator hyper-parameters
        ft_params = self._flatten_nested_dict(params['clf__ft_params'])
        ## Update pre_layer hyper-parameter
        ft_params['clf__pre_layer'] = self.estimator.get_params()['clf__pre_layer'] + add_dense*self.len_dense
        self._estimator_ft = clone(self.estimator, safe=True)
        self._estimator_ft.set_params(**ft_params)
        ## Print and Write verbose
        print("FT Hyper-params:")
        print(ft_params)
        print("FT params:")
        print(self._estimator_ft.get_params()['clf'])
        with open(self.verbose_file, 'a') as f:
            f.write("\nFT Hyper-params:\n")
            f.write(str(ft_params))
            f.write("\nFT params:\n")
            f.write(str(self._estimator_ft.get_params()['clf']))
        ## Fit and Evaluate model
        res_cv = cross_validate(self._estimator_ft, X, y, scoring=self.scoring, cv=self.cv, 
                                return_train_score=self.train_score)
        ## Write verbose
        with open(self.verbose_file, 'a') as f:
            f.write("\nTest AUC: " + str(np.mean(res_cv['test_auc'])))
            f.write("\nTest Sens: " + str(np.mean(res_cv['test_sens'])))
            f.write("\nTest Spec: " + str(np.mean(res_cv['test_spec'])))
            
        return {'loss': 1. - np.mean(res_cv['test_' + self.opt_metric]), 'params': params, 'test_score': res_cv, 
                'status': STATUS_OK}
    
    
    def fit(self, X, y):
        """
        Overrides parent class method.
        
        """

        # Estimate best hyper-params using X, y
        self.best_trial_ = self.model_selection(X, y)
        
        # Instantiate FT estimator model with optimized hyper-params and then fit it
        self._estimator_ft = clone(self.estimator, safe=True)
        self._estimator_ft = self._estimator_ft.set_params(**self._flatten_nested_dict(self.best_trial_['result']['params']['clf__ft_params'])) 
        self.best_estimator_ = self._estimator_ft.fit(X, y)
        
        return self
