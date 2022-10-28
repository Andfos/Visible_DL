import tensorflow as tf
import keras.layers

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
                 term_size_map):
        
        super(RestrictedNN, self).__init__()

        
        self.root = root
        self.n_inp = n_inp
        self.module_neurons = module_neurons
        self.term_direct_gene_map = term_direct_gene_map
        self.layers = layers
        self.dG = dG
       

        self.cal_term_dim(term_size_map)
        # Construct the input layer.
        self.build_input_layer()
        
        # Construct the layers between modules.
        self.build_module_layers(self.dG)


        #self.forward()    
    

    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):

        self.term_dim_map = {}

        for term, term_size in term_size_map.items():
            num_output = int(self.module_neurons)
            
            # log the number of hidden variables per each term
            num_output = int(num_output)
            print("term\t%s\tterm_size\t%d\tnum_hiddens\t%d" % (term, term_size, num_output))
            self.term_dim_map[term] = num_output










    def build_input_layer(self):
        """ Construct the input layer for genotype data."""
        
        self.layers = {}
        inputs = Input(shape=(self.n_inp,))
        
        #with self.name_scope:
        for module, input_set in self.term_direct_gene_map.items():
            self.layers[module] = (Dense(
                    len(input_set),
                    input_shape=(self.n_inp,),
                    activation="linear"))

    
    def build_module_layers(self, dG):
        

        self.term_layer_list = []   # term_layer_list stores the built neural network 
        self.term_neighbor_map = {}


        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)
        
        # Define the leaves of the ontology as those which are not
        # directed towards any other modules.
        leaves = [n for n,d in dG.out_degree() if d==0]

        self.term_layer_list.append(leaves)
    
        for term in leaves:
            
            # input size will be #chilren + #genes directly annotated by the term
            input_size = 0
            
            for child in self.term_neighbor_map[term]:
                input_size += self.term_dim_map[child]
        
            if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])
            

            # term_hidden is the number of the hidden variables in each state
            term_hidden = self.term_dim_map[term]            

            raise



    def forward(self):
        x = np.array([[2, 2, 2]])
        
        for mod, input_set in self.term_direct_gene_map.items():
            
            print(input_set)
            layer = self.layers[mod]

            out = (layer)(x)
            
            print(out)
            raise

            







