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
term_neurons_func = "4*n**2"
regl0 = 0.02
reg_glasso = 0.02
lr = .001


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





for epoch in range(1, 100000):
    
    # calculate the loss over the training set.
    # Get the gradients of all trainable vars and store in a list.
    with tf.GradientTape() as t:

        

        train_preds = res_nn(X_train)
        loss_fn = tf.keras.losses.mean_squared_error
        loss_fn = tf.keras.losses.MeanSquaredError()
        total_loss = loss_fn(train_preds, y_train)
        trainable_vars = res_nn.trainable_variables
        grads = t.gradient(total_loss, trainable_vars)
        

        print(f"Epoch: {epoch}\t\t\t\tLoss: {total_loss}")
        del t


    # Update the trainable variables.
    for i, var in enumerate(trainable_vars):
        var_name = var.name
        var_val = var.value

        # Apply l0 regularization to weights in the input layer.
        if "inp" in var_name and "kernel" in var_name:
            

            # First update the weight by gradient descent.
            #dW = t.gradient(total_loss, var)
            dW = grads[i]
            var.assign_sub(lr * dW)
            
            # Perform proximal l0 regularization.
            c = tf.constant(regl0 * lr)
            new_value = proximal_l0(var, c)
            #var.value = new_value
            var.assign(new_value)
            

        elif "mod" in var_name and "kernel" in var_name:
            var_val = tf.constant(var.numpy())


            mod_name = var_name.split("_")[0].replace("-", ":")
            children = res_nn.mod_neighbor_map[mod_name]
            
            si = 0
            if len(children) != 0:
                #new_value = tf.zeros(var_val.shape)
                new_value = np.zeros(var_val.shape)
                

                for child in children:
                    child_neurons = res_nn.module_dimensions[child]
                    ei = si + child_neurons
                    dW = grads[i][si:ei, :]
                    child_weights = var_val[si:ei, :]
                    

                    
                    # Update the weights by gradient descent.
                    child_weights = child_weights - (lr * dW)

                    # Perform group lasso on the weights coming from a 
                    # child module.
                    child_weights = (proximal_glasso_nonoverlap(
                            child_weights, 
                            reg_glasso*lr))
                    
                
                    new_value[si:ei, :] = child_weights
                
                    # Set the new starting index to the old ending index.
                    si = si + ei
                





                var.assign(new_value)

        # Update bias weights using standard gradient descent.
        else:
            dW = grads[i]
            var.assign_sub(lr * dW)











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

