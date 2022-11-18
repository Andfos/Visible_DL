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


















### Set Parameters ###
# Set general parameters
test_num = "03"
pretrained_model = False
model_infile = "Trained_models/model3"  # Only if using pretrained model.
model_outfile = "Trained_models/pruned_model3" 
prune_train_iterations = 20

# Set the functions and parameters to generate the train and test data.
func = "x[0] + x[1]"
noise_sd = "0"
data_size = 1000
input_dim = 3
lower = -10
upper = 10
test_size = 0.20


# Set neural network parameters
term_neurons_func = "1"
regl0 = 1
reg_glasso = 5 
lr = .01
lip = 0.01
loss_fn = tf.keras.losses.MeanSquaredError()

batch_size = 400
train_epochs = 1000
ngene = 3
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
    train_network(res_nn, X_train, y_train, 
                  train_epochs=train_epochs, loss_fn=loss_fn, optimizer=optimizer)
    raise




    res_history = res_nn.fit(
            X_train, y_train, 
            epochs = train_epochs, batch_size = batch_size, 
            validation_split = 0.2)





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
for prune_train_iter in range(0, prune_train_iterations):

    print(f"Pruning/Training iteration: {prune_train_iter}")
    # Retrain the model.
    res_nn.fit(
            X_train, y_train, 
            epochs = 10, batch_size = batch_size, 
            validation_split = 0.2)

    # Prune the model for a number of prunning epochs.
    prune(res_nn, X_train, y_train, 
          lr=lr, lip=lip, regl0=regl0, reg_glasso=reg_glasso, 
          prune_epochs = 10)

    # Check the network architecture.
    check_network(res_nn, dG, root)



raise








"""
    # Get the gradients of all trainable vars and store in a list.
    with tf.GradientTape() as t:

        # Pass the training data through and make predictions.
        # Calculate the loss.
        #train_preds, mod_output_map = res_nn(X_train, prune=True)
        train_preds = res_nn(X_train, prune=False)
        total_loss = loss_fn(train_preds, y_train)
        

        print(f"Epoch: {epoch}\t\t\t\tLoss: {total_loss}")
        # Get the gradients for all trainable variables.
        trainable_vars = res_nn.trainable_variables
        grads = t.gradient(total_loss, trainable_vars)


        # Delete the gradient tape.
        del t

    # Update the model parameters using PALM.
    res_nn = train_with_palm(res_nn, trainable_vars, grads, lr, regl0, reg_glasso) 
    #check_network(res_nn, dG, root)
# Check network structure.
check_network(res_nn, dG, root)


res_history = res_nn.fit(
        X_train, y_train, 
        epochs = epochs, batch_size = batch_size, 
        validation_split = 0.2)




# Train the network for a certain number of epochs.
for epoch in range(0, 8000):
    
    # Get the gradients of all trainable vars and store in a list.
    with tf.GradientTape() as t:

        
        # Pass the training data through and make predictions.
        # Calculate the loss.
        #train_preds, mod_output_map = res_nn(X_train, prune=True)
        train_preds = res_nn(X_train, prune=False)
        total_loss = loss_fn(train_preds, y_train)
        
        total_loss = 0
        for name, output in mod_output_map.items():
            if name == root:
                added_loss= loss_fn(output, y_train)
                total_loss += added_loss
                print(f"Loss added from {name}: {added_loss}")


            else: # change 0.2 to smaller one for big terms
                added_loss = 0.2 * loss_fn(output, y_train)
                total_loss += added_loss
                print(f"Loss added from {name}: {added_loss}")

        print(total_loss)


        print(f"Epoch: {epoch}\t\t\t\tLoss: {total_loss}")
        # Get the gradients for all trainable variables.
        trainable_vars = res_nn.trainable_variables
        grads = t.gradient(total_loss, trainable_vars)


        # Delete the gradient tape.
        del t

    # Update the model parameters using PALM.
    res_nn = train_with_palm(res_nn, trainable_vars, grads, lr, regl0, reg_glasso) 
    #check_network(res_nn, dG, root)
    

# Check network structure.
check_network(res_nn, dG, root)
"""













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

