import keras
import sys 
from tensorflow.keras import initializers
import datetime
import tensorflow as tf
import numpy as np
from utils import *
#import DrugCell
#from DrugCell import *
from networks import RestrictedNN, save_model
from tensorflow.keras.utils import plot_model
from packaging import version
import tensorboard
from penalties import *
from training import train_with_palm, check_network, prune, train_network
import networkx as nx

















### Set Parameters ###
# Set general parameters
test_num = "04"
pretrained_model = False
model_infile = "Trained_models/model3"  # Only if using pretrained model.
model_outfile = "Trained_models/pruned_model3" 
prune_train_iterations = 20000

# Set the functions and parameters to generate the train and test data.
func = "x[0] * x[1] + x[2]"
noise_sd = "0"
data_size = 1000
input_dim = 7
lower = -10
upper = 10
test_size = 0.20


# Set neural network parameters
term_neurons_func = "16"
regl0 = 5
reg_glasso = 5 
lr = .001
lip = 0.0001
loss_fn = tf.keras.losses.MeanSquaredError()

batch_size = 400
train_epochs = 300
ngene = 7
#initializer = initializers.Ones()
initializer = initializers.GlorotUniform()
input_regularizer = "l2"
module_regularizer = "l2"



optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam')










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
gene2id_mapping = load_mapping(f"Data/geneID_{test_num}.txt")



dG, root, term_size_map, term_direct_gene_map = load_ontology(
        f"Data/onto_{test_num}.txt", 
        gene2id_mapping)



#draw_graph(dG, root, term_size_map, draw_inputs=True, jitter=True)












# Load a pretrained model or initialize a new one.
if pretrained_model:
    res_nn = keras.models.load_model(model_infile)

else:
    res_nn = RestrictedNN(
            root=root, 
            dG=dG,
            module_neurons_func=term_neurons_func, 
            n_inp=ngene, 
            term_direct_gene_map=term_direct_gene_map,
            mod_size_map=term_size_map, 
            initializer=initializer, 
            input_regularizer=input_regularizer,
            module_regularizer=module_regularizer) 


# Compile and build the model.
res_nn.compile(loss='mean_squared_error', optimizer = "adam")
res_nn.build(input_shape = (batch_size, ngene))
res_nn.summary()


#mask = res_nn.build_input_layer(input_regularizer)

res_init_weights = res_nn.get_weights()


# Fit the model if not trained yet. Save it if it converges to solution.
if not pretrained_model:
    res_nn = train_network(res_nn, X_train, y_train, 
                  train_epochs=train_epochs, loss_fn=loss_fn, optimizer=optimizer)



    """
    res_history = res_nn.fit(
            X_train, y_train, 
            epochs = train_epochs, batch_size = batch_size, 
            validation_split = 0.2)
    """




    """
    res_final_weights = res_nn.get_weights()
    print("Initial weights")
    print(res_init_weights)
    print("\n")
    print("Final weights")
    print(res_final_weights)
    print("Mask")

    mask = res_nn.build_input_layer(input_regularizer)
    print(mask)
    dog = res_nn.get_weights()
    print("dof")
    print(dog)
    raise
    


    # Save the model if it converged.
    model_savefile = model_outfile
    train_rmse = round(sqrt(res_nn.evaluate(X_train, y_train)), 2)
    save_model(model_savefile, res_nn, train_rmse, rmse_threshold = 2)

    raise
    """





res_final_weights = res_nn.get_weights()
print("Initial weights")
print(res_init_weights)
print("\n")
print("Final weights")
print(res_final_weights)



### Pruning
# Train the network for a certain number of epochs.

#print(f"Learning rate: {sess.run(optimizer._lr)}")
#current_lr = optimizer._decayed_lr(tf.float32)
#print(current_lr)

test_preds = res_nn(X_test)
loss = loss_fn(y_true=y_test, y_pred=test_preds)
title = f"Trained ontology: Post training, Loss {loss}"

draw_graph(
        dG, root, term_size_map, 
        title=title, draw_inputs=True, jitter=True)


dG_copy = dG.copy()
for prune_train_iter in range(0, prune_train_iterations):
    

    print(f"Pruning/Training iteration: {prune_train_iter}")

    # Prune the model for a number of prunning epochs.
    prune(res_nn, X_train, y_train, 
          lr=lr, lip=lip, regl0=regl0, reg_glasso=reg_glasso,
          inp_id_map=gene2id_mapping,
          prune_epochs=10, debug=False)

    # Retrain the model.
    res_nn = train_network(res_nn, X_train, y_train, 
                  train_epochs=2, loss_fn=loss_fn, optimizer=optimizer)
    
    # Check the network architecture
    dG_copy2 = check_network(res_nn, dG_copy, root, inp_id_map = gene2id_mapping)
    dG_prune = dG_copy2.subgraph(nx.shortest_path(dG_copy2.to_undirected(), root))
    #if (dG_copy.number_of_nodes() != dG_prune.number_of_nodes() 
    #        or dG_copy.number_of_edges() != dG_prune.number_of_edges()):
    if dG_copy.number_of_edges() != dG_copy2.number_of_edges(): 


        for var in optimizer.variables():
            var.assign(tf.zeros_like(var))


        print("Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))
        print("Pruned graph has %d nodes and %d edges" % (dG_prune.number_of_nodes(), dG_prune.number_of_edges()))
        #dG_copy = dG_copy2.copy()
        

        # Retrain the model.
        res_nn = train_network(res_nn, X_train, y_train, 
                      train_epochs=10000, loss_fn=loss_fn, optimizer=optimizer)
        
        test_preds = res_nn(X_test)
        loss = loss_fn(y_true=y_test, y_pred=test_preds)
        title = f"Trained ontology: Iteration {prune_train_iter}, Loss {loss}"
        draw_graph(
                dG_copy, root, term_size_map, 
                title=title, draw_inputs=True, jitter=True)
        draw_graph(
                dG_prune, root, term_size_map, 
                title=title, draw_inputs=True, jitter=True)

        dG_copy = dG_copy2.copy()





res_final_weights = res_nn.get_weights()
print("Final weights")
print(res_final_weights)










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

