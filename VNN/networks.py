import tensorflow as tf
import keras.layers
from tensorflow import keras

import numpy as np
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




from keras.layers import Input



class RestrictedNN():
    def __init__(self, 
                 root, 
                 dG, 
                 module_neurons, 
                 n_inp, 
                 term_direct_gene_map,
                 mod_size_map):
        
        super(RestrictedNN, self).__init__()

        
        self.root = root
        self.n_inp = n_inp
        self.module_neurons = module_neurons
        self.term_direct_gene_map = term_direct_gene_map
        #self.layers = layers
        self.dG = dG
       

        self.get_module_dimensions(mod_size_map)
        # Construct the input layer.
        self.build_input_layer()
        
        # Construct the layers between modules.
        self.build_module_layers(self.dG)


        self.forward()    
         

    def get_module_dimensions(self, mod_size_map):
        """Retrieve the mapping of neurons to modules."""
        self.module_dimensions = {}
       
        for mod, mod_size in mod_size_map.items():
            num_output = int(self.module_neurons)
            
            print(f"Module\t{mod}\tMod_size\t{mod_size}\tNeurons\t{num_output}")
            self.module_dimensions[mod] = num_output





    def build_input_layer(self):
        """ Construct the input layer for genotype data."""
        
        self.gene_layers = {}
        inputs = Input(shape=(self.n_inp,))
        
        # Iterate through the modules that are directly mapped to the input.
        for module, input_set in self.term_direct_gene_map.items():

            
            self.gene_layers[module] = (Dense(
                    len(input_set),
                    input_shape=(self.n_inp,),
                    activation="linear"))

    
    def build_module_layers(self, dG):
        
        self.module_layers = {}
        self.mod_layer_list = []   # term_layer_list stores the built neural network 
        self.mod_neighbor_map = {}

        # Iterate through the nodes of the directed graph. 
        for mod in dG.nodes():
            self.mod_neighbor_map[mod] = []
            
            # Iterate through the children of the node.
            for child in dG.neighbors(mod):
                self.mod_neighbor_map[mod].append(child)
        
        # Define the leaves of the ontology as those which are not
        # directed towards any other modules.
        
        while True:
            leaves = [n for n,d in dG.out_degree() if d==0]


            if len(leaves) == 0:
                break



            self.mod_layer_list.append(leaves)
            
            for mod in leaves:
                
                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0
                
                for child in self.mod_neighbor_map[mod]:
                    input_size += self.module_dimensions[child]
            
                if mod in self.term_direct_gene_map:
                        input_size += len(self.term_direct_gene_map[mod])
                

                # term_hidden is the number of the hidden variables in each state
                mod_hidden = self.module_dimensions[mod]            
                
                # Add a layer for each module.
                self.module_layers[mod] = Dense(
                        mod_hidden, input_shape=(input_size,),
                        activation="linear")

            dG.remove_nodes_from(leaves)
                 



    def forward(self):
        X = np.array([[2, 2, 2, 2],
                      [1, 1, 1, 1]])
        
        # Initialize a dictionary to store output from the first module 
        # layer where input is mapped directly to the module.
        inp_mod_output = {}

        # Iterate through the modules that are directly mapped to the input.
        for mod, input_set in self.term_direct_gene_map.items():
            
            # Pass the entire input vector into the directly-mapped module 
            #layer.
            layer = self.gene_layers[mod]

            # Store the output of each of the directly-mapped module layers as 
            # a separate tensor in a dictionary.
            inp_mod_output[mod] = (layer)(X) 
            
        mod_output_map = {}
        for i, layer in enumerate(self.mod_layer_list):
            print(i)
            print(layer)
            for mod in layer:
                child_input_list = []
                
                print(mod)
                # If the module is directly mapped to other modules, include 
                # the output of the child module in the parent module's input
                # vector.
                for child_mod in self.mod_neighbor_map[mod]:
                    child_input_list.append(mod_output_map[child_mod])
                
                # If the module is directly mapped to the input, include the 
                # output of the first layer in the module's input vector.
                if mod in self.term_direct_gene_map:
                    child_input_list.append(inp_mod_output[mod])

                # Concatenate all of the inputs from child modules together 
                # along the same dimension.
                child_input = tf.concat(child_input_list, 1)
                
                # Pass the input to the module's layer.
                layer = self.module_layers[mod]
                mod_weighted_input = (layer)(child_input) 
                
                # Apply the sigmoid function on the weighted input to the module 
                # neuron. Add the output to a dictionary mapping the term to
                # it's post-activation output.
                mod_output = tf.math.sigmoid(mod_weighted_input)
                mod_output_map[mod] = mod_output
        

        print(mod_output_map["GO:output"])


            

            

            







