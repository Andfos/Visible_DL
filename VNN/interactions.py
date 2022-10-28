from math import sqrt
from tensorflow import keras
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
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
from utils import create_connections_matrix
from plots import plot_preds, plot_loss, plot_preds_3D, generate_3D_plot_data
from networks import create_dense_nn, create_restricted_nn, get_layer_output, save_model
from tensorflow.keras.optimizers import Adam
from gradients import get_gradients
import tempfile





def generate_data(input_size, input_dim, noise, lower, upper):
    """ Create the Input and output """

    X = np.zeros(shape = (input_size, input_dim))
    y = np.zeros(shape = (input_size, 1))

    for i in range(0, input_size):
        
        input_list = np.random.uniform(lower, upper, size = input_dim)
        
        output = input_list[0] * input_list[1] * input_list[2] #+ 3 * input_list[2]\
                 #+ np.random.normal(0, 2*abs(input_list[0]), 1)

        
        X[i] = input_list
        y[i] = output

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2,
                                                        random_state = 42)

    return X_train, X_test, y_train, y_test





if __name__ == "__main__":
    

    ### Set parameters
    # Dense NN parameters
    d_layers = 1
    d_units = [9]
    d_activation = ["sigmoid"]
    d_bias = [True]
    #dense_label = "$Dense: L1_{4}^{lin,bias}L2_{4}^{sig}L3_{1}^{lin}$"
    dense_label = "$Dense: L1_{9}^{sig,bias}L2_{1}^{lin}$"
    
    # Restricted NN parameters
    r_layers = 1
    r_units = [8]
    r_activation = ["sigmoid"]
    r_bias = [True]
    #res_label = "$Res: L1_{20}^{lin,bias}L2_{4}^{sig}L3_{1}^{lin}$"
    res_label = "$Dense: L1_{8}^{sig,bias}L2_{1}^{lin}$"
    
    
    # Data parameters
    function = "y = x[0] * x[1] * x[3]"
    upper = 10
    lower = -10
    noise = 5
    input_dim = 3
    input_size = 1000

    #Neural network parameters
    early_stopping = False
    epochs = 40000
    batch_size = 16
    optimizer = "adam"
    regularizer = None
        
    # Directory for all files generated during script
    directory = ("Interactions_3")
    basename = (f"L{d_layers}_N{d_units}_A{d_activation}_B{d_bias}")

    # Generate connections matrix
    connections = create_connections_matrix("design_matrix.xlsx")

    # Callbacks
    monitor_val_accuracy = EarlyStopping(
            monitor = "val_loss", patience = 1000, min_delta = 3)
    callbacks = [monitor_val_accuracy]






    # Loop through many random weight initializations
    for seed in range(0, 100):
        print("seed {seed}".format(seed = seed))        
        initializer = initializers.GlorotUniform(seed = seed)
        np.random.seed(seed)

        # Generate X and y
        X_train, X_test, y_train, y_test = generate_data(
                input_size, 
                input_dim,
                noise, 
                lower, 
                upper)

        # Initialize the neural networks
        dense_nn = create_dense_nn(
                d_layers, 
                d_units, 
                input_dim=input_dim, 
                activation=d_activation,
                use_bias=d_bias,
                kernel_initializer=initializer)
        
        res_nn = create_restricted_nn(
                r_layers, 
                r_units,
                connections,
                input_dim = input_dim, 
                activation=r_activation,
                use_bias=r_bias,
                kernel_initializer=initializer)
        

        # Get the initial weights
        dense_init_weights = dense_nn.get_weights()
        res_init_weights = res_nn.get_weights()


        # Fit the neural networks
        print("Training dense NN")
        dense_history = dense_nn.fit(
                X_train, y_train, 
                epochs=epochs, batch_size=batch_size,
                validation_split=0.2, 
                callbacks=callbacks)
        
        print("Training restricted NN")
        res_history = res_nn.fit(
                X_train, y_train, 
                epochs = epochs, batch_size = batch_size, 
                validation_split = 0.2, 
                callbacks=callbacks) 
        

        # Get model parameters after training
        dense_weights = dense_nn.get_weights()
        res_weights = res_nn.get_weights()
        """dense_grads = get_gradients(dense_nn, x, y, res_nn.trainable_weights)
        res_grads = get_gradients(res_nn, x, y, res_nn.trainable_weights)"""



        # Evaluate the model on the train and test set
        dense_train_rmse = round(sqrt(dense_nn.evaluate(X_train, y_train)), 2)
        dense_test_rmse = round(sqrt(dense_nn.evaluate(X_test, y_test)), 2)
        
    
        res_train_rmse = round(sqrt(res_nn.evaluate(X_train, y_train)), 2)
        res_test_rmse = round(sqrt(res_nn.evaluate(X_test, y_test)), 2)
    

        # Save the models if they converged
        dense_saveFile = f"{directory}/Trained_models/Dense_{basename}_{seed}"
        res_saveFile = f"{directory}/Trained_models/Res_{basename}_{seed}"
        converge_d = save_model(
                dense_saveFile, dense_nn, dense_train_rmse, rmse_threshold = 22)
        
        
        converge_r = save_model(
                res_saveFile, res_nn, res_train_rmse, rmse_threshold = 22)
        
        
        # Record information if either of the models converge
        if converge_d == True  or converge_r == True:
            converge = True
        else:
            converge = False
        
        """
        # Generate 3D plot data
        x1, x2, X_modelInp = generate_3D_plot_data(
                lower = -10, upper = 10, size = 100)
        
        fx = x1 * x2
        

        # Make predictions for the plot data
        dense_plotPreds = dense_nn.predict(X_modelInp)
        dense_plotPreds = dense_plotPreds.reshape(100, 100)


        # Get the output of the L-1 layer
        layer_out = get_layer_output(dense_nn, "dense_1", X_modelInp)
        dense_weights = dense_nn.get_weights()
        """
        

        """
        # Get the weighted input to the first node of the L-1 layer
        preAct_node = -1 * np.log(1 / (layer_out[:,0]) - 1) * 10  # Mult by 10
        # to scale up
        preAct_node = preAct_node.reshape(100, 100)
        """

        
        
        


        """
        # Compute the weighted input to the output node from the first node in
        # L-1 layer

        # Get the weighted input to thehj
        for i in range(len(layer_out)):
            for j in range(len(layer_out[0])):
                layer_out[i][j] = layer_out[i][j]*dense_weights[2][j]


        weighted_inputs = []
        for i in range(len(layer_out[0])):
            a_w = layer_out[:, i].reshape(100,100)
            weighted_inputs.append(a_w)
        """

        """
        # Plot the predictions
        plot_preds_3D(
                f"{directory}/Images/predictions_{basename}_{seed}_{converge}",
                function, x1, x2, fx,
                converge=converge, 
                Dense = [dense_plotPreds, dense_test_rmse, "blue", dense_label],
                #Res = [res_plotPreds, res_test_rmse, "red", res_label])
        """
        
        
        
        # Plot the loss curve
        plot_loss(
                f"{directory}/Images/loss_{basename}_{seed}_{converge}",
                train_size = len(X_train), 
                converge=converge, 
                Dense_train = [dense_history.history["loss"], dense_train_rmse,
                               dense_label + "; train"],
                Dense_test = [dense_history.history["val_loss"], dense_test_rmse,
                              dense_label + "; test"],
                Res_train = [res_history.history["loss"], res_train_rmse, 
                             res_label + "; train"],
                Res_test = [res_history.history["val_loss"], res_test_rmse, 
                            res_label + "; test"])
                


       
        # Record the initial and final weights
        weights_file = f"{directory}/Images/{basename}_weights_{seed}.txt"
        
        if converge == True:
            with open(weights_file, "w") as outf:
                outf.write("Dense 4 4 initial weights\n{}\n\n".format(dense_init_weights))
                outf.write("Dense 4 4 final weights\n{}\n\n".format(dense_weights))
                outf.write("Res 2 2 initial weights\n{}\n\n".format(res_init_weights))
                outf.write("Res 2 2  final weights\n{}\n\n".format(res_weights))


################################################################################
        """
        # Plot the weighted inputs
        if converge == True:
            plot_preds_3D(
                    f"{directory}/Images/{basename}_weightedInputs_{seed}_{converge}",
                    function, x1, x2, fx,
                    converge = converge, 
                    Dense = [dense_plotPreds, dense_test_rmse, "blue",
                             dense_label], 
                    a1 = [weighted_inputs[0], "NA", "red", "$a_{1}^{2} * w_{1,1}^{3}$"], 
                    a2 = [weighted_inputs[1], "NA", "yellow", "$a_{2}^{2} * w_{1,2}^{3}$"], 
                    a3 = [weighted_inputs[2], "NA", "green", "$a_{3}^{2} * w_{1,3}^{3}$"], 
                    a4 = [weighted_inputs[3], "NA", "purple", "$a_{4}^{2} * w_{1,4}^{3}$"]) 
        """
