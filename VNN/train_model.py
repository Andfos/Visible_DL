import sys 
from tensorflow.keras import initializers
import datetime
import tensorflow as tf
import numpy as np
from utils import *
#import DrugCell
#from DrugCell import *
from networks import RestrictedNN
from tensorflow.keras.utils import plot_model
from packaging import version
import tensorboard



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
batch_size = 16
epochs = 5000
term_neurons = 4
ngene = 3
initializer = initializers.Ones()



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
        module_neurons=term_neurons, 
        n_inp=ngene, 
        term_direct_gene_map=term_direct_gene_map,
        mod_size_map=term_size_map, 
        initializer=initializer)

res_nn.compile(loss='mean_squared_error', optimizer = "adam")
res_nn.build(input_shape = (batch_size, ngene))
res_nn.summary()


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

