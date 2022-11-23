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

            # Maybe delete this.
            G.add_edge(parent, child)

    file_handle.close()
    n_inp = len(input_set)

    # Iterate through the modules in the directed graph.
    for module in G.nodes():
        
        # If the module is an input, skip it.
        if not module[0].isupper():
            module_size_map[module] = 1
            continue

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









def draw_graph(G, root, module_size_map, 
               title="Ontology", draw_inputs=True, jitter=False):

    G_copy = G.copy()
    color_list = []
    node_list = []
    size_list = []
    pos_map = {}
        
    x = 0.0
    while True:

        leaves = [n for n,d in G_copy.out_degree() if d==0]
        
        

        if len(leaves) == 0:
            break
        
        if len(leaves) % 2 != 0:
            mid = int((len(leaves)-1)/2)
        else:
            mid = int(len(leaves)/2)
        

        
        for i, node in enumerate(leaves):
            y = (mid - i) * 3

            
            if jitter and not node[0].islower():
                x_coord = x + float(np.random.uniform(-0.2, 0.2, 1))
                
                if y > 0:
                    y_coord  =  y + float(np.random.uniform(0, 0.5, 1))

                elif y < 0:
                    y_coord  =  y + float(np.random.uniform(-0.5, 0, 1))
                
                else:
                    y_coord  =  y + float(np.random.uniform(0, 0, 1)) 


                pos_map[node] = (x_coord, y_coord)
            else:
                pos_map[node] = (x, y)


            if node == root:
                node_list.append(node)
                color_list.append("lightblue")
                size_list.append(200 * module_size_map[node] + 600)

            elif node[0].isupper():
                node_list.append(node)
                color_list.append("coral")
                size_list.append(200 * module_size_map[node] + 600)
            
            else:
                if draw_inputs == True:
                    node_list.append(node)
                    color_list.append("lightgreen")
                    size_list.append(600)


        x += 1.0
        G_copy.remove_nodes_from(leaves) 
        

    nx.draw_networkx(G, 
            arrows = True, 
            node_shape="o", 
            pos=pos_map,
            nodelist=node_list,
            node_color=color_list, 
            node_size = size_list,
            font_size=6)

    plt.title(title)
    plt.show()

    



















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







def debugger(mode, run_debugger = True, **kwargs):

    if mode == "input_layer" and run_debugger == True:
        var_name = kwargs["var_name"]
        init_val = kwargs["init_val"]
        lr = kwargs["lr"]
        dW = kwargs["dW"]

        print(f"\nUpdating {var_name}...\n")
        new_var = init_val - lr*dW
        print("Initial weights - (lr * gradient) = New weights")
        for b in range(len(init_val)):
            print(f"{init_val[b]}\t-\t{lr} * {dW[b]}\t=\t{init_val[b]}\n")
        print("\n")            

    # Debugging for proximal L0 (penalties.py)
    if mode == "L0" and run_debugger == True:
        alpha_abs = kwargs["alpha_abs"]
        theta = kwargs["theta"]
        alpha_new = kwargs["alpha_new"]

        alpha_array = np.array(alpha_abs.numpy())
        print("Abs(Weights) >< Theta = New weights")
        for b in range(len(alpha_array)):
            print(f"{alpha_abs[b]}\t>\t{theta}\t=\t{alpha_new[b]}\n")
        print("\n\n")

    # Debugging for weight updates between the input layer and a module layer
    # directly mapped to an input layer.
    if mode == "module_layer" and run_debugger == True:
        child_name = kwargs["child_name"]
        mod_name = kwargs["mod_name"]
        init_weights = kwargs["init_weights"]
        lr = kwargs["lr"]
        dW = kwargs["dW"]
        temp_weights = kwargs["temp_weights"]


        
        print(f"\nUpdating weights from {child_name} that feed ", 
              f"into {mod_name}...\n")
        
        print("Initial weights - (lr * gradient) = New weights")
        if init_weights.ndim > 1:
            for dim in range(init_weights.ndim):
                print(f"{init_weights[dim]}\t-\t{lr} * {dW[dim]}\t=\t{temp_weights[dim]}")
            print("\n")            
        
        else:
            print(f"{init_weights}\t-\t{lr} * {dW}\t=\t{temp_weights}")
            print("\n")            
    
    if mode == "group_lasso" and run_debugger == True:
        alpha = kwargs["alpha"]
        alpha_norm = kwargs["alpha_norm"]
        c = kwargs["c"]
        alpha_new = kwargs["alpha_new"]

    
        print("If matrix norm < c, matrix --> 0")
        print("Otherwise, matrix --> (matrix / matrix_norm) * (matrix_norm - c)")

        print(f"Matrix norm: {alpha_norm} <> C: {c}")
        if alpha.ndim > 1:
            for dim in range(alpha.ndim):
                print(f"{alpha[dim]}\t-->\t{alpha_new[dim]}")
            print("\n\n")
        else:
            print(f"{alpha}\t-->\t{alpha_new}")
            print("\n\n")






