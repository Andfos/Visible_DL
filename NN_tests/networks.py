import tensorflow as tf
import numpy as np
import keras.backend as K
from tensorflow.keras import layers
from keras.layers import Dense, Layer, Input, Concatenate, Add
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model








class RestrictedLayer(Dense):
    """ Restricted NN Layer """
    def __init__(self, units, connections, **kwargs):
        
        # This refers to the matrix of 1's and 0's that specify the connections
        self.connections = connections
        super().__init__(units, **kwargs)

    def call(self, inputs):
        """ Dense implements the operation: output = activation(dot(input,
        kernel) + bias)"""
        
        # Multiply the weights (kernel) element-wise with the connections
        # matrix. Then take the dot product of the input with the
        # weighted connections.
        output = K.dot(inputs, self.kernel * self.connections)
        
        # If the bias is to be included, add it with the output of the previous
        # step.
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is None:
            raise ValueError("Activation required")
        output = self.activation(output)
        return output







def create_dense_nn(layers, 
                    units, 
                    input_dim = 1, 
                    activation = "linear",
                    optimizer = "adam",
                    kernel_initializer = "glorot_uniform", 
                    kernel_regularizer = None, 
                    trainable = True, 
                    use_bias = False):
    
    """ Dense (Fully-Connected) Model """
    model = Sequential()
    for i in range(layers):
        model.add(Dense(units[i], 
                        input_shape = (input_dim,), 
                        activation = activation[i], 
                        use_bias = use_bias[i], 
                        kernel_initializer = kernel_initializer, 
                        kernel_regularizer = kernel_regularizer, 
                        name = f"dense_{i+1}"))

    model.add(Dense(1, activation = "linear", use_bias = False,
                    kernel_initializer = kernel_initializer,
                    trainable = trainable,
                    name = "Output"))
    
    model.compile(loss='mean_squared_error', optimizer = optimizer)
    model.summary()

    return model






def create_restricted_nn(layers, 
                         units, 
                         connections, input_dim = 1, 
                         activation = "sigmoid",
                         optimizer = "adam",
                         kernel_initializer = "glorot_uniform", 
                         kernel_regularizer = None, 
                         trainable = True, 
                         use_bias = False):
    
    """ Restricted NN Model """
    model = Sequential()
    for i in range(layers):
        model.add(RestrictedLayer(units[i], 
                                  connections[i],  
                                  input_shape = (input_dim,),
                                  activation = activation[i], 
                                  use_bias = use_bias[i], 
                                  kernel_initializer = kernel_initializer, 
                                  kernel_regularizer = kernel_regularizer, 
                                  name = f"res_{i+1}"))
    model.add(Dense(1, activation = "linear", use_bias = False, 
                    kernel_initializer = kernel_initializer, 
                    trainable = trainable, 
                    name = "Output"))
    model.compile(loss='mean_squared_error', optimizer = optimizer)
    model.summary()

    return model








# Created by ebanner (2016)
def get_gradients(model):
    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    weights = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)

















def get_layer_output(model, layer_name, X):
  
    layer_output=model.get_layer(layer_name).output  #get the Output of the Layer
    intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output) 
    layer_out = intermediate_model.predict(X)
    
    return layer_out




def save_model(filepath, model, model_rmse, rmse_threshold = 1):
    """ Checks if model has converged to solution. Model is saved if it has"""
    converge = False
    if model_rmse <= rmse_threshold:
        converge = True
        model.save(filepath)

    return converge









