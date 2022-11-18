import tensorflow as tf
import keras.layers
from tensorflow import keras
from tensorflow.keras import initializers
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
from utils import set_module_neurons



from keras.layers import Input









class RestrictedLayer(Dense):
    """ Build a layer where a specific connections matrix can be applied.  """
    
    def __init__(self, units, connections, **kwargs):
        
        # This refers to the matrix of 1's and 0's that specify the connections
        super().__init__(units, **kwargs)
        self.connections = connections
        #self.kernel_regularizer = input_regularizer


    def call(self, inputs):
        """ Dense implements the operation: output = activation(dot(input,
        kernel) + bias)"""
        
        # Multiply the weights (kernel) element-wise with the connections
        # matrix. Then take the dot product of the input with the
        # weighted connections.
        output = K.dot(inputs, self.kernel * self.connections)

        # If the bias is to be included, add it with the output of the previous
        # step.
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        
        # Apply the activation function.
        if self.activation is None:
            raise ValueError("Activation required")
        output = self.activation(output)
        
        # Return the ouput of the layer.
        return output

















class RestrictedNN(tf.keras.Model):
    """ Creates a neural network where connections between modules can be 
    specified."""
    
    def __init__(self, 
                 root, 
                 dG, 
                 module_neurons_func, 
                 n_inp, 
                 term_direct_gene_map,
                 mod_size_map, 
                 initializer,
                 input_regularizer,
                 module_regularizer):
        
        super(RestrictedNN, self).__init__()

        
        self.root = root
        self.n_inp = n_inp
        self.module_neurons_func = module_neurons_func
        self.term_direct_gene_map = term_direct_gene_map
        #self.layers = layers
        self.dG = dG       
        self.initializer = initializer
        #self.input_regularizer = input_regularizer
        #self.module_regularizer = module_regularizer

        self.get_module_dimensions(mod_size_map)
        #self.mask = self.create_connections_mask()
        #print(self.mask)
        #raise
        # Construct the input layer.
        self.build_input_layer(input_regularizer)
        
        # Construct the layers between modules.
        self.build_module_layers(self.dG, module_regularizer)
        



    def get_module_dimensions(self, mod_size_map):
        """Retrieve the number of neurons allocated to  each  module layers.
         
        The number of neurons in each module layer is passed in as function.
        It can be static, where each module layer gets the same number of 
        inputs, or it can be dynamic, where the number of neurons is a 
        function of the number of inputs to the layer.
        """
        
        self.module_dimensions = {}
        self.module_children_num = {}
        # Iterate through the modules of the ontology.
        
        print("Module\tModule_size\tModule_direct_children\tNeurons")
        for mod, mod_size in mod_size_map.items():
            
            # Set the number of neurons in each module layer according to the 
            #module_neurons_func.
            n = 0
            if mod == self.root:
                n = 1
                num_children = len(self.dG[mod])
            elif mod in self.term_direct_gene_map:
                n += len(self.term_direct_gene_map[mod])
                num_children = n
            else:
                n += len(self.dG[mod])
                num_children = n
            
     
            num_neurons = set_module_neurons(n, self.module_neurons_func)
            self.module_children_num[mod] =  num_children 
            self.module_dimensions[mod] = num_neurons

            print(f"{mod}\t{mod_size}\t{num_children}\t{num_neurons}")


    
    def create_connections_mask(self):

        
        mask = {}
        for module, input_set in self.term_direct_gene_map.items():
            input_set = sorted(list(input_set))
            
            # Construct the connections matrix.
            connections = np.zeros( (self.n_inp, len(input_set)) )
            j = 0
            for i in input_set:
                connections[i][j] = 1
                j+=1
            
            mask[module] = connections
        
        return mask
        


    def build_input_layer(self, input_regularizer):
        """ Construct the input layer for each input-connected module.

        The name of the input layer follows the form <module_name>_inp. The
        layer is passed the entire vector of inputs, and it returns only the 
        inputs that belong to the specified module. There are as many neurons 
        in the layer as there are inputs to the module, with the output of 
        each neuron representing an exact mapping to exactly one input. 
        For instance, if module1 takes input from inp1 and inp2, the input 
        layer will be named module1_inp, it will have 2 neuons, and the output 
        of neuron 1 will equal the value of input 1, and the output of neuron 2 
        will equal the value of input 2. The weights are initialized to 1 and
        are untrainable, and the activation is linear.
        """
        
        # Initialize a dictionary to keep track of input layers for each 
        # module.
        self.gene_layers = {}
        self.direct_layers = {}
        # Iterate through the modules that are directly mapped to the input.
        for module, input_set in self.term_direct_gene_map.items():
            input_set = sorted(list(input_set))
            
            # Construct the connections matrix.
            connections = np.zeros( (self.n_inp, len(input_set)) )
            j = 0
            for i in input_set:
                connections[i][j] = 1
                j+=1
            
            
            # Create a restriced layer with untrainable weights initialized 
            # to 1 and a linear activation.
            mod_name = f"{module.replace(':', '-')}_inp"
            mod_name_direct = f"{module.replace(':', '-')}_direct"
            
            self.gene_layers[module] = (RestrictedLayer(
                    units=len(input_set),
                    connections=connections,
                    input_shape=(self.n_inp,),
                    activation="linear",
                use_bias=False,
                name=mod_name,
                #kernel_initializer=initializers.Ones(), 
                kernel_initializer=initializers.Ones(),
                #kernel_regularizer=input_regularizer,
                trainable=False))
            

            self.direct_layers[module] = (RestrictedLayer(
                    units=len(input_set),
                    connections=np.identity(len(input_set)),
                    input_shape=(len(input_set),),
                    activation="linear",
                use_bias=False,
                name=mod_name_direct,
                kernel_initializer=initializers.Ones(),
                kernel_regularizer=input_regularizer,
                trainable=True))




    def build_module_layers(self, dG, module_regularizer):
        """ Construct the layers for each module in the ontology.

        The module layer takes in a tensor output from a module_inp layer if 
        it is directly mapped to inputs, or it takes in a tensor that is
        created by concatenating the tensors output from all of the module's 
        child modules. By default, the sigmoid activation is applied and a bias 
        term is included. 
        """
        # Make a copy of the graph.
        dG_copy = dG.copy()

        # Initialize a dictionary to store the layers of the modules.
        self.module_layers = {}
        self.mod_layer_list = []   # term_layer_list stores the built neural network 
        self.mod_neighbor_map = {}

        # Iterate through the nodes of the directed graph. 
        for mod in dG_copy.nodes():
            self.mod_neighbor_map[mod] = []
            
            # Iterate through the children of the node.
            for child in dG_copy.neighbors(mod):
                self.mod_neighbor_map[mod].append(child)
        
        # Define the leaves of the ontology as those which are not
        # directed towards any other modules.
        
        while True:
            leaves = [n for n,d in dG_copy.out_degree() if d==0]

           
            if len(leaves) == 0:
                break



            self.mod_layer_list.append(leaves)
            

            for mod in leaves:
                
                # Input size will be the number of children plus the number of 
                # inputs directly mapped to the module.
                #input_size = 0
                #input_size = self.module_children_num[mod]
                input_size = 0
                
                for child in self.mod_neighbor_map[mod]:
                    input_size += self.module_dimensions[child]
            
                # If the module is directly mapped to inputs, make sure the 
                # layer name reflects this for later use.
                if mod in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[mod])
                
                mod_hidden = self.module_dimensions[mod]            
                
                # Add a layer for each module.
                #with self.name_scope:
                

                mod_name = f"{mod.replace(':', '-')}_mod"
                mod_output = f"{mod}-output"
                mod_output_name = f"{mod_name}_output"
                if mod == self.root:
                    self.module_layers[mod] = Dense(
                            1, input_shape=(input_size,),
                            activation="linear", 
                            name=mod_name, 
                            use_bias=False, 
                            kernel_initializer=self.initializer,
                            kernel_regularizer=module_regularizer)
                else:
                    self.module_layers[mod] = Dense(
                            self.module_dimensions[mod], 
                            input_shape=(input_size,),
                            activation="linear", 
                            name=mod_name, 
                            use_bias=False, 
                            kernel_initializer=self.initializer,
                            kernel_regularizer=module_regularizer)
                    
                                         
                    # Add a layer with a single neuron to represent the final
                    # output of the module.
                    self.module_layers[mod_output] = Dense(
                            1, 
                            input_shape=(self.module_dimensions[mod],),
                            activation="linear", 
                            name=mod_output_name, 
                            use_bias=False, 
                            kernel_initializer=self.initializer)
                    
                
            dG_copy.remove_nodes_from(leaves)
                 


    def call(self, inputs, prune=False):
        #X = np.array([[2, 2, 2, 2],
        #              [1, 1, 1, 1]])
        
        #X = np.array([[2, 2, 2, 2]]).astype("float64")

        # Initialize a dictionary to store output from the first module 
        # layer where input is mapped directly to the module.
        inp_mod_output = {}

        # Iterate through the modules that are directly mapped to the input.
        for mod, input_set in self.term_direct_gene_map.items():
            
            # Pass the entire input vector into the directly-mapped module 
            #input layer.
            inp_layer = self.gene_layers[mod]
            direct_layer = self.direct_layers[mod]
            
            # Store the output of each of the directly-mapped module layers as 
            # a separate tensor in a dictionary.
            temp = inp_layer(inputs)
            inp_mod_output[mod] = direct_layer(temp) 


        
        mod_output_map = {}
        for i, layer in enumerate(self.mod_layer_list):
            for mod in layer:
                child_input_list = []
                # If the module is directly mapped to other modules, include 
                # the output of the child module in the parent module's input
                # vector.

                for child_mod in self.mod_neighbor_map[mod]:

                    child_mod_output_name = f"{child_mod}-output"
                    
                    child_input_list.append(mod_output_map[child_mod])
                    #child_input_list.append(mod_output_map[child_mod_output_name])
                

                # If the module is directly mapped to the input, include the 
                # output of the first layer in the module's input vector.
                if mod in self.term_direct_gene_map:
                    child_input_list.append(inp_mod_output[mod])
                
                # Concatenate all of the inputs from child modules together 
                # along the same dimension.
                child_input = tf.concat(child_input_list, 1)


                # Pass the input to the module's layer.
                layer = self.module_layers[mod]
                mod_weighted_input = layer(child_input) 
                

                # If there are no auxiliary layers...
                mod_output_map[mod] = mod_weighted_input
                
                
                """
                if mod == self.root:
                    mod_output_map[mod] = mod_weighted_input

                else:

                    mod_output_name = mod_output = f"{mod}-output"
                    output_layer = self.module_layers[mod_output_name]
                    mod_output = output_layer(mod_weighted_input)

                    # Apply the sigmoid function on the weighted input to the module 
                    # neuron. Add the output to a dictionary mapping the term to
                    # it's post-activation output.
                    
                    #mod_output = tf.math.sigmoid(mod_weighted_input)
                    #mod_output_map[mod] = mod_weighted_input
                    mod_output_map[mod_output_name] = mod_output
                """


        if prune == False:
            return(mod_output_map[self.root])
        
        else:
            return(mod_output_map[self.root], mod_output_map)


            

            


def save_model(filepath, model, model_rmse, rmse_threshold = 1):
    """ Checks if model has converged to solution. Model is saved if it has."""
    converge = False
    if model_rmse <= rmse_threshold:
        print("Model saved.")
        converge = True
        model.save(filepath)

    return converge




















            




class MLP(tf.Module):
  def __init__(self, input_size, sizes, name=None):
    super(MLP, self).__init__(name=name)
    self.layers = []
    with self.name_scope:
      for size in sizes:
        self.layers.append(Dense(input_dim=input_size, output_size=size))
        input_size = size
  @tf.Module.with_name_scope
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


