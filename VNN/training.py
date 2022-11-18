import tensorflow as tf
import numpy as np
from penalties import *
import networkx as nx





def train_network(model, X_train, y_train, train_epochs, loss_fn, optimizer):
    
    # Get the connections mask.
    mask = model.create_connections_mask()
    
    # Iterate for a number of training epochs.
    for epoch in range(train_epochs):
        
        # Pass the training data through the model and make predictions.
        # Compute the loss, including the loss from regularization.
        # Get the gradients for all trainable variables.
        
        with tf.GradientTape() as tape:
            train_preds = model(X_train)
            loss = loss_fn(y_true=y_train, y_pred=train_preds)
            loss += sum(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        del tape
        
        # Uodate the weights of the variables in the nwtwork.
        for i, var in enumerate(model.trainable_variables):
            module_name = var.name.split("_")[0].replace("-",":")
            
            # Apply the mask to the input gradients.
            if "direct" in var.name:
                grads[i] = grads[i] * mask[module_name]
            

        
        print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(),
                                          loss.numpy()))
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(model.trainable_variables)

    




















def train_with_palm(
        model, trainable_vars, grads, lr, lip, regl0, reg_glasso, debug=False):
    
    # Update the trainable variables.
    for i, var in enumerate(trainable_vars):
        var_name = var.name


        # Apply l0 regularization to weights in the input layer.
        if "direct" in var_name and "kernel" in var_name:

            # First update the weight by gradient descent.
            dW = grads[i]
            var.assign_sub(lip * dW)
            
            # Print calculations if debug mode is turned on.
            if debug:
                print(f"\nUpdating {var_name}...\n")
                init_val = np.array(var.numpy())
                new_var = init_val - lr*dW
                print("Initial weights\t-\tlr   *   gradient\t=\tNew weights")
                for b in range(len(init_val)):
                    print(f"{init_val[b]}\t-\t{lr} * {dW[b]}\t=\t{init_val[b]}\n")
                print("\n")            
            
            # Perform proximal l0 regularization.
            c = tf.constant(regl0 * lip)
            new_value = proximal_l0(var, c, debug=True)
            var.assign(new_value)
            
        # Apply the group lasso to the weights between modules.
        elif "mod" in var_name and "kernel" in var_name:
            var_val = tf.constant(var.numpy())
            mod_name = var_name.split("_")[0].replace("-", ":")
            
            

            #print(f"Currently updating the module {mod_name}")



            # Iterate through the modules that input to the current module. 
            children = model.mod_neighbor_map[mod_name]
            si = 0
            if len(children) != 0:
                
                # Initialize an array to record the new weights of the module.
                new_value = np.zeros(var_val.shape)
                
                for child in children:
                    child_neurons = model.module_dimensions[child]
                    
                    ei = si + child_neurons
                    #ei = si + 1
                    child_weights = var_val[si:ei, :]
                    
                    print(f"""Updating the weights from {child} that feed into {mod_name}""")
                    print(f"Initial weights:\n{child_weights}\n")
                    

                    # Update the weights that come from the specific child 
                    #module by gradient descent.
                    dW = grads[i][si:ei, :]
                    child_weights = child_weights - (lr * dW)
                    


                    print(f"Gradients:\n{dW}\n")
                    print(f"Final weights:\n{child_weights}\n")
                    print("Performing group lasso...")
                    # Perform group lasso on the weights coming from a 
                    # child module.
                    child_weights = (proximal_glasso_nonoverlap(
                            child_weights, 
                            reg_glasso*lr))

                    # Pass the new weights into the array of 0s.
                    new_value[si:ei, :] = child_weights
                
                    # Set the new starting index to the old ending index.
                    si = si + ei
                
                # Update the variables weights.
                var.assign(new_value)
                print(var)

        # Update bias weights using standard gradient descent.
        else:
            dW = grads[i]
            var.assign_sub(lr * dW)

    # Return the model with updated parameters.
    return model




def check_network(model,dG, root):
    dG_copy = dG.copy()
    

    trainable_vars = model.trainable_variables
    for i, var in enumerate(trainable_vars):
        var_name = var.name
        


        if "inp" in var_name and "kernel" in var_name:
            print("Input layer\n")
            print(f"{var_name}\t\t\t\t{var.value}\n")
            
            

        # Apply l0 regularization to weights in the input layer.
        if "mod" in var_name and "kernel" in var_name:
            var_val = tf.constant(var.numpy())
            mod_name = var_name.split("_")[0].replace("-", ":")
            

            # Iterate through the modules that input to the current module. 
            children = model.mod_neighbor_map[mod_name]
            si = 0
            if len(children) != 0:
                
                # Initialize an array to record the new weights of the module.
                new_value = np.zeros(var_val.shape)
                
                for child in children:
                    
                    child_neurons = model.module_dimensions[child]
                    ei = si + child_neurons
                    #ei = si + 1

                    child_weights = var_val[si:ei, :]
                    nonzero_weights = np.count_nonzero(child_weights)
                    
                    if nonzero_weights == 0:
                        dG_copy.remove_edge(mod_name, child)


                    print("Module layers\n")
                    print(f"Input from {child} to {mod_name}")
                    print(child_weights)
                    

                    # Set the new starting index to the old ending index.
                    si = si + ei



    print("Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))
    sub_dG_prune = dG_copy.subgraph(nx.shortest_path(dG_copy.to_undirected(), root))
    print("Pruned graph has %d nodes and %d edges" % (sub_dG_prune.number_of_nodes(), sub_dG_prune.number_of_edges()))






def prune(
        model, 
        X_train, y_train, 
        lr, lip, regl0, reg_glasso, 
        prune_epochs = 10, 
        debug=True):
    """ Input the model with the training data, return the pruned model."""
    
    # Define the loss function.
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Prune the model for a specified number of prune_epochs.
    for prune_epoch in range(prune_epochs):
        with tf.GradientTape() as t:

            # Pass the training data through and make predictions.
            # Calculate the loss.
            train_preds = model(X_train, prune=False)
            total_loss = loss_fn(train_preds, y_train)
            print(f"Prune Epoch: {prune_epoch}\t\tLoss: {total_loss}")
            
            # Get the gradients for all trainable variables.
            trainable_vars = model.trainable_variables
            grads = t.gradient(total_loss, trainable_vars)
            del t

        # Update the model parameters using PALM.
        model = train_with_palm(model, trainable_vars, grads, 
                                lr=lr, lip=lip, regl0=regl0, reg_glasso=reg_glasso, 
                                debug=True) 
    

    return model




