from math import sqrt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from math import sqrt
import tensorflow as tf
import keras.backend as K
from tensorflow.keras import layers
from keras.layers import Dense, Layer, Input, Concatenate, Add
from keras.models import Sequential
#from keras import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from utils import create_connections_matrix
from plots import plot_preds, plot_loss
from networks import create_dense_nn, create_restricted_nn
from tensorflow.keras.optimizers import Adam
#from networks import get_gradients
from gradients import get_gradients
import tempfile





def generate_data(input_size, input_dim, noise):
    """ Create the Input and output """

    X = np.zeros(shape = (input_size, input_dim))
    y = np.zeros(shape = (input_size, 1))

    for i in range(0, input_size):
        #input_list = np.random.randint(low = lower, high = upper, size = input_dim)
        input_list = [np.random.uniform(lower, upper)]
        output = input_list[0] ** 2 #+ np.random.normal(0, noise, 1)

        X[i] = input_list
        y[i] = output

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2,
                                                        random_state = 42)

    return X_train, X_test, y_train, y_test





if __name__ == "__main__":
    

    ### Set parameters
    # Data parameters
    function = "y = X**2" #+ E ~ N(0,5)"
    upper = 5
    lower = -5
    noise = 5
    input_dim = 1
    input_size = 1000

    #Neural network parameters
    early_stopping = False
    epochs = 5000
    batch_size = 16
    optimizer = "adam"
    #optimizer = Adam(beta_1 = 0.2)


    activation = "sigmoid"
    initializer = initializers.Ones()
    #initializer = "glorot_uniform"
    trainable = True
    regularizer = None
        
    

    # Generate connections matrix
    connections = create_connections_matrix("design_matrix.xlsx")

    for i in range(68,100):
        print("seed {}".format(i))        
        initializer = initializers.GlorotUniform(seed = i)
        #initializer = initializers.Ones()

        np.random.seed(i)
        #tf.set_random_seed(i)

        ### Fit models to data
        # Generate X and y
        X_train, X_test, y_train, y_test = generate_data(
                input_size, 
                input_dim,
                noise)

        # Initialize the neural networks
        dense_nn = create_dense_nn(2, 
                                   [2,2], 
                                   input_dim = input_dim, 
                                   activation = ["linear", "sigmoid"],
                                   use_bias = [True, False],
                                   kernel_initializer = initializer)
        
       
        
        res_nn = create_restricted_nn(2, 
                                      [2,2],
                                      connections,
                                      input_dim = input_dim, 
                                      activation = ["linear", "sigmoid"],
                                      use_bias = [True, False],
                                      kernel_initializer = initializer)
        """


        
        res_nn = create_restricted_nn(2, 
                                      [3,3], 
                                      connections, 
                                      input_dim = input_dim, 
                                      activation = activation, 
                                      kernel_initializer = initializer)
        """

        # Get the initial weights
        dense_init_weights = dense_nn.get_weights()
        res_init_weights = res_nn.get_weights()

        # Fit a linear model and the neural networks
        reg = LinearRegression().fit(X_train, y_train)

        res_history = res_nn.fit(X_train, y_train, 
                                     epochs = epochs, batch_size = batch_size, 
                                     #validation_split = 0.2, 
                                     validation_data = (X_test, y_test))
        dense_history = dense_nn.fit(X_train, y_train, 
                                     epochs = epochs, batch_size = batch_size, 
                                     #validation_split = 0.2, 
                                     validation_data = (X_test, y_test))
        
        

        """
        x = np.array([[1],[2],[3]])
        y = np.array([[1],[4],[9]])
        x = np.array([[6]])
        y = np.array([[36]])

        #x = y = np.random.randn(32, 1)
        #dense_nn.train_on_batch(x, y)
        print("These are the weights")
        print(res_nn.get_weights())
        print("\n\n\n\n")
        grads = get_gradients(res_nn, x, y, res_nn.trainable_weights)
        print(grads)
        raise


        weights = [tensor for tensor in res_nn.trainable_weights]
        print(weights)
        optimizer = res_nn.optimizer
        print(res_nn.compute_loss(3))
        raise
        #a = optimizer.get_gradients(res_nn.Model.total_loss, weights)
        print(a)
        raise
        grads = get_gradients(res_nn)
        raise
        """
        """
        inp = res_nn.input
        outputs = [layer.output for layer in res_nn.layers]          # all layer outputs
        functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
        test = [5]
        test = np.random.random(1)[np.newaxis]
        #[np.newaxis,...]
        print(test)
        layer_outs = [func([test]) for func in functors]
        print(layer_outs)
        raise

        layer_name = "res_0"
        layer_output = res_nn.get_layer(layer_name).output
        print(layer_output)
        print(dir(layer_output))
        print(layer_output.shape)
        print(layer_output.node)
        print(layer_output.name)
        print(layer_output[0])
        raise
        """
        
        """
        tf.compat.v1.disable_eager_execution() 





        outputTensor = res_nn.output #Or model.layers[index].output
        



        print(outputTensor)
        listOfVariableTensors = res_nn.trainable_weights
        print(listOfVariableTensors)
        gradients = K.gradients(outputTensor, listOfVariableTensors)
        print(gradients)
        raise
        """

        # Evaluate the model on the train and test set
        dense_train_rmse = round(sqrt(dense_nn.evaluate(X_train, y_train)), 2)
        dense_test_rmse = round(sqrt(dense_nn.evaluate(X_test, y_test)), 2)
        res_train_rmse = round(sqrt(res_nn.evaluate(X_train, y_train)), 2)
        res_test_rmse = round(sqrt(res_nn.evaluate(X_test, y_test)), 2)

        converge = "N"
        if dense_train_rmse <= 3 or res_train_rmse <= 3:
            converge = "Y"
            if dense_train_rmse <= 1:
                dense_nn.save("Trained_Models/dense_2Lin_2Sig")
            if res_train_rmse <= 1:
                res_nn.save("Trained_Models/res_2Lin_2Sig")


        # Create the X and y variables for plotting f(x)
        X_plot = np.linspace(-10, 10, 1000)
        y_plot = X_plot**2 

        # Fit a linear regression model on X and y
        reg = LinearRegression().fit(X_train, y_train)

        # Make predictions using the fitted neural networks and linear model on X
        # plotting variable
        reg_pred = reg.predict(X_plot.reshape(-1, 1))
        dense_predicts = dense_nn.predict(X_plot)
        res_predicts = res_nn.predict(X_plot)
        dense_predicts = [item for sublist in dense_predicts for item in sublist]
        res_predicts = [item for sublist in res_predicts for item in sublist]

        # Plot the predictions
        plot_preds("Images/L2_2NLin_2NSig/predictions_{}_{}".format(i, converge),
                   function, 
                   X_plot,          
                   y_plot, 
                   Linear_regression = [reg_pred, "NA"], 
                   Dense_2_2 = [dense_predicts, dense_test_rmse], 
                   Res_2_2 = [res_predicts, res_test_rmse])
        
        # Plot the loss curve
        plot_loss("Images/L2_2NLin_2NSig/loss_{}_{}".format(i, converge),
                  train_size = len(X_train), 
                  Dense_2_2_train = [dense_history.history["loss"], dense_train_rmse],
                  Dense_2_2_test = [dense_history.history["val_loss"], dense_test_rmse],
                  Res_2_2_train = [res_history.history["loss"], res_train_rmse],
                  Res_2_2_test = [res_history.history["val_loss"], res_test_rmse])

        # Print the initial and final weights
        dense_weights = dense_nn.get_weights()
        res_weights = res_nn.get_weights()
        

        weights_file = "Images/L2_2NLin_2NSig/weights_{}.txt".format(i)
        with open(weights_file, "w") as outf:
            outf.write("Dense 2 2 initial weights\n{}\n\n".format(dense_init_weights))
            outf.write("Dense 2 2 final weights\n{}\n\n".format(dense_weights))
            outf.write("Res 2 2 initial weights\n{}\n\n".format(res_init_weights))
            outf.write("Res 2 2  final weights\n{}\n\n".format(res_weights))


        print("Dense initial weights\n{}\n\n".format(dense_init_weights))
        print("Dense final weights\n{}\n\n".format(dense_weights))
        print("Restricted initial weights\n{}\n\n".format(res_init_weights))
        print("Restricted final weights\n{}\n\n".format(res_weights))
