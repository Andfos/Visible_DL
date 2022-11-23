import tensorflow as tf
import numpy as np
from penalties import *
import networkx as nx
from utils import debugger




def train_network(model, X_train, y_train, train_epochs, loss_fn, optimizer):
    
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
            
            #Apply the gradient mask to the gradient matrix.
            mod_name = var.name.split("/")[0]
            
            if "kernel" in var.name:
                grads[i] = grads[i] * model.grad_masks[mod_name]

        # Update the variables of the model using the optimizer.
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(),
                                          loss.numpy()))

    return model
    








def train_with_palm(
        model, trainable_vars, grads, lr, lip, regl0, reg_glasso, 
        inp_id_map, debug=False):
    
    # Update the trainable variables.
    for i, var in enumerate(trainable_vars):
        var_name = var.name
        layer_name = var_name.split("/")[0]
        grad_mask = model.grad_masks[layer_name]



        #1). Apply l0 regularization to weights in the input layer.
        if "inp" in var_name and "kernel" in var_name:

            # First update the weight by gradient descent.
            init_val = np.array(var.numpy())  # For debugging only.
            dW = grads[i] * grad_mask
            var.assign_sub(lip * dW)
            
            # Print calculations if debug mode is turned on.
            if debug:
                debugger("input_layer", True,
                         var_name=var_name, init_val=init_val, 
                         lr=lip, dW=dW)

            # Perform proximal l0 regularization.
            c = tf.constant(regl0 * lip)
            new_value = proximal_l0(var, c, debug=debug)
            var.assign(new_value)
    



        #2). Apply the group lasso to the weights between modules.
        elif "mod" in var_name and "kernel" in var_name:
            var_val = tf.constant(var.numpy())
            mod_name = var_name.split("_")[0].replace("-", ":")
            

            #2a). If the module is directly mapped to inputs, apply group lasso
            # on the neuron from the input layer that represents the gene 
            # only. 
            if mod_name in model.term_direct_gene_map:
                new_value = np.zeros(var_val.shape)
                
                # Iterate across all of the neurons from the module's input
                # layer. Apply gradient descent.
                for index, id in enumerate(model.term_direct_gene_map[mod_name]):
                    
                    inp = list(inp_id_map.keys())[id]
                    init_input_weights = var_val[index, :]
                    dW = grads[i][index, :] * grad_mask[index, :]
                    temp_input_weights = init_input_weights - (lip * dW)
                    
                    if debug:
                        debugger("module_layer", True, 
                                 child_name=inp, mod_name=mod_name,
                                 init_weights=init_input_weights,
                                 lr=lip, dW=dW, temp_weights=temp_input_weights)
                        
                    # Apply group lasso to the weights from a single input.
                    final_input_weights = (proximal_glasso_nonoverlap(
                            temp_input_weights, 
                            reg_glasso*lip, debug=debug))
                    new_value[index, :] = final_input_weights
                
                # Assign the module the newly calculated weights.
                var.assign(new_value)
                


            #2b). If the module is directly mapped to other modules, apply 
            # group lasso on the weights from each of the individual child 
            # modules.
            children = model.mod_neighbor_map[mod_name]
            si = 0
            
            if len(children) != 0:
                new_value = np.zeros(var_val.shape)
                
                # Iterate through each of the child modules.
                for child in children:
                    
                    child_neurons = model.module_dimensions[child]
                    ei = si + child_neurons
                    init_child_weights = var_val[si:ei, :]
                    dW = grads[i][si:ei, :] * grad_mask[si:ei, :]
                    temp_child_weights = init_child_weights - (lip * dW)
                    
                    if debug:
                        debugger("module_layer", True, 
                                 child_name=child, mod_name=mod_name,
                                 init_weights=init_child_weights,
                                 lr=lip, dW=dW, temp_weights=temp_child_weights)
                    # Perform group lasso on the weights coming from a 
                    # child module.
                    final_child_weights = (proximal_glasso_nonoverlap(
                            temp_child_weights, 
                            reg_glasso*lip, debug=debug))
                    new_value[si:ei, :] = final_child_weights
                
                    # Set the new starting index to the old ending index.
                    si = si + ei
                
                # Update the variables weights.
                var.assign(new_value)

        
            """
            var_val = tf.constant(var.numpy())
            dW = grads[i] * grad_mask
            temp_weights = var_val - (lr * dW)
            c = tf.constant(.0001)
            final_weights = proximal_l2(temp_weights, c)
            var.assign(final_weights)
            """
        #3). Update other weights using regular gradient descent.
        else:
            dW = grads[i]
            var.assign_sub(lr * dW)
            c = tf.constant(.001)
            new_bias = proximal_l2(var, c)
            var.assign(new_bias)
        


        # Update the gradient mask if a weight went to 0.
        zero_weights = (new_value == 0)
        try:
            zero_weights = zero_weights.numpy()
        except AttributeError:
            pass
        grad_mask[zero_weights] = 0
        model.grad_masks[layer_name] = grad_mask
    
    # Return the model with updated parameters.
    return model






def check_network(model,dG_copy, root, inp_id_map):
    
    dG_copy2 = dG_copy.copy()
    trainable_vars = model.trainable_variables
    for i, var in enumerate(trainable_vars):
        var_name = var.name
        var_val = np.array(var.numpy())  
        mod_name = var_name.split("_")[0].replace("-", ":")
        

        # Check which weights have gone to 0.
        if "kernel" in var_name:
            
            if mod_name in model.term_direct_gene_map:

                # Remove the connection between genes
                for index, id in enumerate(model.term_direct_gene_map[mod_name]):
                    if np.count_nonzero(var_val[index] == 0) > 0:
                        inp = list(inp_id_map.keys())[id]
                        
                        try:
                            dG_copy2.remove_edge(mod_name, inp)
                        except:
                            pass
            
            else:
                children = model.mod_neighbor_map[mod_name]
                si = 0
                
                # Iterate through each of the child modules.
                for child in children:
                    
                    child_neurons = model.module_dimensions[child]
                    ei = si + child_neurons
                    child_weights = var_val[si:ei, :]
                    
                    if np.count_nonzero(child_weights == 0) > 0:
                        inp = list(inp_id_map.keys())[id]
                        
                        try:
                            dG_copy2.remove_edge(mod_name, child)
                        except:
                            pass
                    # Update the starting index.
                    si = si + ei

    return dG_copy2
   

"""
        elif "mod" in var_name and "kernel" in var_name:
            if mod_name in model.term_direct_gene_map:
                for index, id in enumerate(model.term_direct_gene_map[mod_name]):
                    
                    inp = list(inp_id_map.keys())[id]
        




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

"""





def prune(
        model, 
        X_train, y_train, 
        lr, lip, regl0, reg_glasso, 
        inp_id_map, prune_epochs = 10, 
        debug=False):
    """ Input the model with the training data, return the pruned model."""
    
    # Define the loss function.
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Prune the model for a specified number of prune_epochs.
    for prune_epoch in range(prune_epochs):
        with tf.GradientTape() as t:

            # Pass the training data through and make predictions.
            # Calculate the loss.
            train_preds = model(X_train)
            total_loss = loss_fn(train_preds, y_train)
            
            # Add in the loss from the regularization.
            #print(model.losses)
            #total_loss += sum(model.losses)                 #Maybe delete.
            print(f"Prune Epoch: {prune_epoch}\t\tLoss: {total_loss}")
            
            # Get the gradients for all trainable variables.
            trainable_vars = model.trainable_variables
            grads = t.gradient(total_loss, trainable_vars)
        del t

        # Update the model parameters using PALM.
        model = train_with_palm(model, trainable_vars, grads, 
                                lr=lr, lip=lip, regl0=regl0, reg_glasso=reg_glasso, 
                                inp_id_map=inp_id_map, debug=debug) 
    

    return model
















