""" The script creates a restricted neural network and a dense neural network. 

Both are fit to data generated via either of the two functions:
                        
                    1.    y = sum(x) + N(0,1)
                    2.    y = 2 * x[0] + 3 * x[1] + x[2] + N(0,3)

Early stopping after 5 epochs in which the validation accuracy does not increase 
is implemented by default. The mean squared error is used as a loss, and the 
root mean squared error (irreducible loss = 1) is reported at the end."""

# Import packages
import numpy as np             
import pandas as pd
from math import sqrt
import tensorflow as tf
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




""" Set Parameters """

### Input parameters
noise = 3
train_sizes = [3, 4, 5, 6, 7, 8, 10, 12, 20] 


early_stopping = False
epochs = 10000
batch_size = 2
optimizer = "adam"
activation = "linear"

### Set the kernel initializer
#initializer = initializers.Zeros()
initializer = initializers.Ones()
#initializer = "glorot_uniform"

trainable = True


### Set the regularizer
#regularizer = "l0"
#regularizer = "l1"
#regularizer = "l2"
regularizer = None


### Plot parameters
loss_lim = 200



"""Create the connections matrix to define the connections between input 
and neurons in the restricted layer """
connections_matrix = np.zeros((3, 2))
connections_matrix[0][0] = 1
connections_matrix[1][0] = 1
connections_matrix[2][1] = 1






""" Restricted NN Layer """
class RestrictedLayer(Dense):

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





""" Restricted NN Model """
def create_restricted_nn():
    model = Sequential()
    model.add(RestrictedLayer(2, connections_matrix, 
                              input_shape = (3,), 
                              activation = activation, 
                              use_bias = False, 
                              kernel_initializer = initializer, 
                              kernel_regularizer = regularizer
                              ))
    model.add(Dense(1, activation = activation, use_bias = False, 
                    kernel_initializer = initializer, 
                    trainable = trainable))
    model.compile(loss='mean_squared_error', optimizer = optimizer)
    model.summary()

    return model




""" Dense (Fully-Connected) Model """
def create_dense_nn(units):
    model = Sequential()
    model.add(Dense(units, input_shape = (3,), activation = activation, 
                    use_bias = False, 
                    kernel_initializer = initializer, 
                    kernel_regularizer = regularizer
                    ))
    model.add(Dense(1, activation = activation, use_bias = False,
                    kernel_initializer = initializer,
                    trainable = trainable))
    model.compile(loss='mean_squared_error', optimizer = optimizer)
    model.summary()

    return model





""" Functional API Model """
def create_functional_model():
    num_1 = Input(shape = (1,), name = "num_1")
    num_2 = Input(shape = (1,), name = "num_2")
    num_3 = Input(shape = (1,), name = "num_3")
    
    in_1_2 = Concatenate()([num_1, num_2])
    out_1_2 = Dense(
            1, activation = activation, use_bias = False, 
            kernel_initializer = initializer, kernel_regularizer = regularizer)(in_1_2)
    in_1_2_3 = Concatenate()([out_1_2, num_3])
    output = Dense(
            1, activation = activation, use_bias = False,
            kernel_initializer = initializer, kernel_regularizer = regularizer,
            trainable = trainable)(in_1_2_3)
    

    model = Model([num_1, num_2, num_3], output)
    model.compile(loss = "mean_squared_error", optimizer = optimizer)
    model.summary()
    return model



""" Dense (Fully-Connected) Model """
def two_layer_dense_nn(units):
    model = Sequential()
    model.add(Dense(units, input_shape = (3,), activation = activation, 
                    use_bias = False, 
                    kernel_initializer = initializer, 
                    kernel_regularizer = regularizer
                    ))
    model.add(Dense(2, activation = activation, use_bias = False,
                    kernel_initializer = initializer,
                    trainable = trainable))
    model.add(Dense(1, activation = activation, use_bias = False,
                    kernel_initializer = initializer,
                    trainable = trainable))
    model.compile(loss='mean_squared_error', optimizer = optimizer)
    model.summary()

    return model










""" Function to plot the traiing versus validation loss"""
def plot_loss(res_loss, res_val_loss, 
              dense_loss, dense_val_loss,
              func_loss, func_val_loss,
              res_test_mse, dense_test_mse, func_test_mse,
              size = "100%"):
    
    fig, ax = plt.subplots()

    plt.plot(res_loss)
    plt.plot(res_val_loss)
    plt.plot(dense_loss)
    plt.plot(dense_val_loss)
    plt.plot(func_loss)
    plt.plot(func_val_loss)
    plt.title("Model loss: Train size = {}".format(size))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['2_train', '2_test', 
                '3_train', '3_test', 
                '2_layer_tain', '2_layer_test'], loc='upper right')

    
    textstr = "2 node MSE: {}\n3 node MSE: {}\n2 layer MSE: {}".format(res_test_mse, 
                                                                dense_test_mse, 
                                                                func_test_mse)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.1, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
    ax.set_ylim(0, loss_lim)
    plt.show()


def plot_train_sizes(sizes, 
                     res_train_loss, res_test_loss, 
                     dense_train_loss, dense_test_loss,
                     func_train_loss, func_test_loss):
    plt.figure()
    plt.plot(sizes, res_train_loss, marker = "o")
    plt.plot(sizes, res_test_loss, marker = "o")
    plt.plot(sizes, dense_train_loss, marker = "o")
    plt.plot(sizes, dense_test_loss, marker = "o")
    plt.plot(sizes, func_train_loss, marker = "o")
    plt.plot(sizes, func_test_loss, marker = "o")
    plt.title("Model loss with training size")
    plt.ylabel("Loss")
    plt.xlabel("Train Size")
    plt.legend(["Res_train", "Res_test", 
                "Dense_train", "Dense_test", 
                "Func_train", "Func_test"])
    plt.show()











""" Create the Input and output """
# Generate input and output arrays
input_size = 1000

X = np.zeros(shape = (input_size, 3))
y = np.zeros(shape = (input_size, 1))

for i in range(0, input_size):
    input_list = np.random.randint(low = 0, high = 10, size = 3)
    

    #output = np.sum(input_list) + np.random.normal(0, noise, 1)
    output = (input_list[0]*2 + input_list[1]*3)*4 + input_list[2] + np.random.normal(0, noise, 1)
    #output = (input_list[0]*2 + input_list[1]*3)*2 - input_list[2] + np.random.normal(0, noise, 1) 
    #output = (input_list[0] + input_list[1])*2 - input_list[2]


    X[i] = input_list
    y[i] = output


# Split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 42)





""" Fit the models and make predictions """

# Instantiate the models
restricted_nn = create_dense_nn(2)
dense_nn = create_dense_nn(4)
#func_nn = create_dense_nn(4)
func_nn =  two_layer_dense_nn(2)



# Plot the models
plot_model(func_nn, to_file = "functional_nn.png")
img = plt.imread("functional_nn.png")
plt.imshow(img)
plt.show()


# Create early stopping callback
monitor_val_accuracy = EarlyStopping(monitor = "val_loss", patience = 5)

if early_stopping == True:
    callbacks = [monitor_val_accuracy]
else:
    callbacks = []





# Get the initial weights of the model prior to training
init_res_weights = restricted_nn.get_weights()
init_dense_weights = dense_nn.get_weights()
init_func_weights = func_nn.get_weights()




# Loop over mulitple training sizes

res_train_loss = []
res_test_loss = []
dense_train_loss = []
dense_test_loss = []



for size in train_sizes:
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]
    
    # Set the weights to the initial weights prior to training
    restricted_nn.set_weights(init_res_weights)
    dense_nn.set_weights(init_dense_weights)
    func_nn.set_weights(init_func_weights)

    # Fit both models
    input_df = pd.DataFrame.from_records(X_train_frac)
    output_df = pd.DataFrame.from_records(y_train_frac)
    input_test_df = pd.DataFrame.from_records(X_test)
    output_test_df = pd.DataFrame.from_records(y_test)
    



    # Fit the functional neural network
    """func_history = func_nn.fit([input_df[0], input_df[1], input_df[2]], y_train_frac, 
                               epochs = epochs, batch_size = batch_size, 
                               #validation_split = 0.2,
                               validation_data = ([input_test_df[0],
                                                   input_test_df[1], 
                                                   input_test_df[2]], 
                                                   output_test_df), 
                               callbacks = callbacks)


        
    func_predicts = func_nn.predict(
            [input_test_df[0], input_test_df[1], input_test_df[2]])
    
    print(func_predicts)
    print("\n\n\n\n\n\n\n\n\n\n\n")
    for i in range(len(func_predicts)):
        print("{} {} {}: {}".format(X_test[i][0], 
                                    X_test[i][1],
                                    X_test[i][2],
                                    y_test[i]))"""


    

    





    # Fit the restricted neural network
    res_history = restricted_nn.fit(X_train_frac, y_train_frac, 
                                    epochs = epochs, batch_size = batch_size, 
                                    #validation_split = 0.2,
                                    validation_data = (X_test, y_test),
                                    callbacks = callbacks)

    # Fit the dense neural network
    dense_history = dense_nn.fit(X_train_frac, y_train_frac, 
                                 epochs = epochs, batch_size = batch_size, 
                                 #validation_split = 0.2, 
                                 validation_data = (X_test, y_test),
                                 callbacks = callbacks)

    
    # Fit the restricted neural network
    func_history = func_nn.fit(X_train_frac, y_train_frac, 
                                    epochs = epochs, batch_size = batch_size, 
                                    #validation_split = 0.2,
                                    validation_data = (X_test, y_test),
                                    callbacks = callbacks)

    # Print the training data
    print("Training input")
    print(X_train_frac)
    print("\nTraining output")
    print(y_train_frac)


    # Print the initial weights
    print("Initial Restricted Weights\n{}\n\n".format(init_res_weights))
    print("Initial Dense Weights\n{}\n\n".format(init_dense_weights))
    print("Initial Func Weights\n{}\n\n\n\n\n".format(init_func_weights))

    # Print the final weights
    print("Final Restricted Weights\n{}\n\n".format(restricted_nn.get_weights()))
    print("Final Dense Weights\n{}\n\n".format(dense_nn.get_weights()))
    print("Final Func Weights\n{}\n\n".format(func_nn.get_weights()))


    # Get the train mse of each model after training
    res_train_mse = restricted_nn.evaluate(X_train_frac, y_train_frac)
    dense_train_mse = dense_nn.evaluate(X_train_frac, y_train_frac)
    func_train_mse = func_nn.evaluate(X_train_frac, y_train_frac)
    
    # Get the test mse of each model after training
    res_test_mse = restricted_nn.evaluate(X_test, y_test)
    dense_test_mse = dense_nn.evaluate(X_test, y_test)
    func_test_mse = func_nn.evaluate(X_test, y_test)
    




    # Print the MSE
    print("Restricted NN Train MSE: {}".format(res_train_mse))
    print("Restricted NN Test MSE: {}".format(res_test_mse))
    print("Dense NN Train MSE: {}".format(dense_train_mse))
    print("Dense NN Test MSE: {}".format(dense_test_mse))
    print("Functional NN Train MSE: {}".format(func_train_mse))
    print("Functional NN Test MSE: {}".format(func_test_mse))

    
    # Plot the training versus validation loss
    plot_loss(res_history.history["loss"], 
              res_history.history["val_loss"], 
              dense_history.history["loss"], 
              dense_history.history["val_loss"],  
              func_history.history["loss"], 
              func_history.history["val_loss"],  
              res_test_mse, 
              dense_test_mse,
              func_test_mse,
              size)
    
    # Append the training and testing mse at each training size
    res_train_loss.append(restricted_nn.evaluate(X_train, y_train))
    res_test_loss.append(restricted_nn.evaluate(X_test, y_test))
    dense_train_loss.append(dense_nn.evaluate(X_train, y_train))
    dense_test_loss.append(dense_nn.evaluate(X_test, y_test))





