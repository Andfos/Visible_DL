import sys 
from tensorflow.keras import initializers
import datetime
import tensorflow as tf
import numpy as np
from utils import *
#import DrugCell
#from DrugCell import *
from networks import RestrictedNN, MLP
from tensorflow.keras.utils import plot_model
from packaging import version
import tensorboard
from penalties import *


### Set Parameters ###
# Set the functions and parameters to generate the train and test data.
func = "x[0] * x[1] + x[2]"
noise_sd = "0"
data_size = 1000
input_dim = 3
lower = -10
upper = 10
test_size = 0.20



# Set neural network parameters
term_neurons_func = "n**2"
regl0 = 0.02
lr = 1


batch_size = 16
epochs = 100
ngene = 3
#initializer = initializers.Ones()
initializer = initializers.GlorotUniform()


### Load Data ###
function_name = f"f(X) = {func} + E~N(0, {noise_sd}"

# Generate X and y vectors.
X, y = generate_data(
        function=func,
        noise_sd_func=noise_sd,
        data_size=data_size,
        input_dim=input_dim,  
        lower=lower,
        upper=upper)

# Split the training and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size,
                                                    random_state=42)


# load ontology
gene2id_mapping = load_mapping("Data/geneID_test.txt")
dG, root, term_size_map, term_direct_gene_map = load_ontology("Data/onto_test2terms.txt", gene2id_mapping)


# Load the VNN
res_nn = RestrictedNN(
        root=root, 
        dG=dG, 
        module_neurons_func=term_neurons_func, 
        n_inp=ngene, 
        term_direct_gene_map=term_direct_gene_map,
        mod_size_map=term_size_map, 
        initializer=initializer) 

print(res_nn)
#print(res_nn.variables)
#print(res_nn.name_scope)
#print(res_nn.submodules)


"""
module = MLP(input_size=5, sizes=[5, 5])
print(module.variables)
raise
"""


res_nn.compile(loss='mean_squared_error', optimizer = "adam")
res_nn.build(input_shape = (batch_size, ngene))
res_nn.summary()





for epoch in range(1, epochs):
    print(f"Epoch {epoch}")
    
    # calculate the loss over the training set.
    with tf.GradientTape() as t:

        train_preds = res_nn(X_train)
        loss_fn = tf.keras.losses.mean_squared_error
        loss_fn = tf.keras.losses.MeanSquaredError()
        total_loss = loss_fn(train_preds, y_train)
        print(total_loss)


        print("\n")

        # Update the trainable variables.
        trainable_vars = res_nn.trainable_variables
        for var in trainable_vars:
            var_name = var.name
            var_val = var.value
            
            # Apply l0 regularization to weights in the input layer.
            if "inp" in var_name and "kernel" in var_name:

                # First update the weight by gradient descent.
                dW = t.gradient(total_loss, var)
                var.assign_sub(lr * dW)
                
                # Perform proximal l0 regularization.
                c = tf.constant(regl0 * lr)
                new_value = proximal_l0(var, c)
                var.value = new_value
                                
            

        


        #print(t.gradient(total_loss, res_nn))
        raise


    raise


raise






















# Get initial weights.
print("Initial weights")
res_init_weights = res_nn.get_weights()
print(res_init_weights)


# Load callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


print("Training restricted NN")
res_history = res_nn.fit(
        X_train, y_train, 
        epochs = epochs, batch_size = batch_size, 
        validation_split = 0.2,
        callbacks=[tensorboard_callback]) 

res_preds = res_nn.predict(X_test)


# Get final weights.
print("Final weights")
res_final_weights = res_nn.get_weights()
print(res_final_weights)

