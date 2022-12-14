import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import *


class drugcell_nn(nn.Module):

    def __init__(self, term_direct_gene_map, dG, ngene, root, term_neurons):
    
        super(drugcell_nn, self).__init__()

        self.root = root
        self.term_neurons = term_neurons
        
        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map   

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(term_size_map)           


        # ngenes, gene_dim are the number of all genes  
        self.gene_dim = ngene              

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.forward()
        raise

        self.construct_NN_graph(dG)
        

        
        # add modules for neural networks to process drugs  
        self.construct_NN_drug()

        self.CUDA_ID = CUDA_ID

        # add modules for final layer
        final_input_size = num_hiddens_genotype + num_hiddens_drug[-1]
        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final,1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))

    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):

        self.term_dim_map = {}
        
        for term, term_size in term_size_map.items():
            num_output = self.term_neurons
                
            # log the number of hidden variables per each term
            num_output = int(num_output)
            print("term\t%s\tterm_size\t%d\tnum_hiddens\t%d" % (term, term_size, num_output))
            self.term_dim_map[term] = num_output


    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):
        
        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no directed asscoiated genes for', term)
                sys.exit(1)
    
            # if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes         
            self.add_module(term+'_direct_gene_layer', nn.Linear(self.gene_dim, len(gene_set), bias = False))



    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):

        self.term_layer_list = []   # term_layer_list stores the built neural network 
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term   
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            #leaves = [n for n in dG.nodes() if dG.in_degree(n) == 0]   
            leaves = [n for n,d in dG.out_degree() if d==0]
            #leaves = [n for n,d in dG.out_degree() if d==0]

            if len(leaves) == 0:
                break

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

                self.add_module(term+'_GO_linear_layer', nn.Linear(input_size, term_hidden, bias = False))
                self.add_module(term+'_GO_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term+'_GO_aux_linear_layer1', nn.Linear(term_hidden,1))
                self.add_module(term+'_GO_aux_linear_layer2', nn.Linear(1,1))

            dG.remove_nodes_from(leaves)
    
    def forward(self):
        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)




 cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
