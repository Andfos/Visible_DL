import sys
from math import *
import numpy as np
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Functions allowed for data generation.
math_funcs = {"abs":abs, "sqrt":sqrt, "log":log, "log10":log10, "exp":exp} 


def generate_data(function,
                  noise_sd_func = 0,
                  data_size = 100, 
                  input_dim = 1,
                  lower = -10, 
                  upper = 10):
    """ 
    Manually generate the input and output data according to user-specified 
    functions..
    """
    X = np.zeros(shape = (data_size, input_dim))
    y = np.zeros(shape = (data_size, 1))

    # Iterate for the desired data size to produce X and y vectors.
    for i in range(0, data_size):
        input_list = np.random.uniform(lower, upper, size = input_dim)
        

        try:
            # Generate output according to the user-specified function.
            math_funcs["x"] = input_list
            output = eval(function,{"__builtins__":None}, math_funcs)
            
            # Add in the noise generated from the specified noise standard
            # deviation.
            noise_sd = eval(noise_sd_func,{"__builtins__":None}, math_funcs)
            noise = np.random.normal(0, noise_sd, 1)
            output += noise
            
        except IndexError:
            print("Ensure that the input_dim matches the dimensions of the "
                  "function.")
            sys.exit()

                
        X[i] = input_list
        y[i] = output


    return X, y





def set_module_neurons(n, dynamic_neurons_func):
    """ Retrieve the number of neurons for a given module. 

    This function allows the user to specify the number of neurons a module 
    will contain as a function of the number of its inputs.
    """
    math_funcs["n"] = n
    mod_neurons = eval(dynamic_neurons_func, {"__builtins__":None}, math_funcs)
    return(mod_neurons)




def load_mapping(mapping_file):

    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    
    return mapping






def load_ontology(file_name, input_id_map):
    """
    Load the ontology file and return a directed acyclic graph to represent
    it.
    """

    # Initialize an empty directed graph and sets
    G = nx.DiGraph()
    module_direct_input_map = {}
    module_size_map = {}

    
    file_handle = open(file_name)
    input_set = set()

    # Iterate through the ontology file.
    for line in file_handle:
        line = line.rstrip().split()
        parent = line[0]
        child = line[1]
        relation = line[2]

        # If mapping between two modules, add to directed graph.
        if relation == 'default':
            G.add_edge(parent, child)
        
        # If the input is not in the input IDs, skip it.
        else:
            if child not in input_id_map:
                continue
            
            # If the module is mapped directly to inputs, instantiate a new 
            # set to record its inputs.
            if parent not in module_direct_input_map:
                module_direct_input_map[parent] = set()

            # Add the id of the input to the module that it is mapped to.
            module_direct_input_map[parent].add(input_id_map[child])
            
            # Add the input to the set of all inputs.
            input_set.add(child)


    file_handle.close()
    n_inp = len(input_set)

    # Iterate through the modules in the directed graph.
    for module in G.nodes():
        module_input_set = set()

        # If the module has a direct mapping to an input, 
        # add the input to the module's input set.
        if module in module_direct_input_map:
            module_input_set = module_direct_input_map[module]
        
        # Get a list of the descendants of the module.
        deslist = nxadag.descendants(G, module)

        # Iterate through the descendents of the module.
        for des in deslist:                         
            
            # Add any genes that are annotated to the child to the parents
            # gene set.
            if des in module_direct_input_map:
                module_input_set = module_input_set | module_direct_input_map[des]
        
        # If any of the terms have no genes in their set, break.
        if len(module_input_set) == 0:
            print("Module {module} is empty. Please delete it.")
            sys.exit(1)
        else:
            module_size_map[module] = len(module_input_set)

    leaves = [n for n,d in G.in_degree() if d==0]

    uG = G.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))


    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print('There are more than connected components. Please connect them.')
        sys.exit(1)

    return G, leaves[0], module_size_map, module_direct_input_map











    



















#gene2id_mapping = load_mapping("gene2ind.txt")
#gene2id_mapping = load_mapping("geneID_test.txt")


#dG, leaves, term_size_map, term_direct_gene_map = load_ontology("drugcell_ont.txt", gene2id_mapping)
#dG, root, term_size_map, term_direct_gene_map = load_ontology("ontology_test.txt", gene2id_mapping)



"""
nx.draw_networkx(dG, 
                 arrows = True, 
                 node_shape="s", 
                 node_color = "white")

plt.title("Organogram of a company.")
plt.show()
"""




def build_input_vector(row_data, num_col, original_features):

    cuda_features = torch.zeros(len(row_data), num_col)

    for i in range(len(row_data)):
        data_ind = row_data[i]
        cuda_features.data[i] = original_features.data[data_ind]
   
    return cuda_features














