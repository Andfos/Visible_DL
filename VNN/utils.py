import sys
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import matplotlib.pyplot as plt


def load_mapping(mapping_file):

    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    
    return mapping






def load_ontology(file_name, gene2id_mapping):
    
    # Initialize an empty directed graph and sets
    dG = nx.DiGraph()
    term_direct_gene_map = {}
    term_size_map = {}

    # Open the file containing the ontology
    file_handle = open(file_name)

    # Add genes to the a set
    gene_set = set()

    for line in file_handle:
        line = line.rstrip().split()
        
        # If the line is a mapping between GO terms, add to directed graph.
        if line[2] == 'default':
            dG.add_edge(line[0], line[1])
        
        # If the gene is not in the gene mappings, skip it.
        else:
            if line[1] not in gene2id_mapping:
                continue
            
            # If the gene has not yet been mapped to a GO term,
            # add the gene to the term mapping.
            if line[0] not in term_direct_gene_map:
                term_direct_gene_map[ line[0] ] = set()

            # Add the mapping between the GO term and the index of the gene.
            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])
            
            # Add the gene to the set of genes.
            gene_set.add(line[1])


    file_handle.close()
    print('There are %d genes' % len(gene_set))

    # Iterate through the GO terms in the directed graph.
    for term in dG.nodes():
        term_gene_set = set()

        # If the GO term has a direct mapping to a gene, 
        # add the gene to the GO term's gene set.
        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]
        
        # Get a list of the descendants of the GO term.
        deslist = nxadag.descendants(dG, term)

        # Iterate through the descendents of the term.
        for child in deslist:                         
            
            # Add any genes that are annotated to the child to the parents
            # gene set.
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]
        
        # If any of the terms have no genes in their set, break.
        if len(term_gene_set) == 0:
            print('There is empty terms, please delete term: %s' % term)
            sys.exit(1)
        else:
            term_size_map[term] = len(term_gene_set)

    leaves = [n for n,d in dG.in_degree() if d==0]

    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    print('There are %d roots: %s' % (len(leaves), leaves[0]))
    print('There are %d terms' % len(dG.nodes()))
    print('There are %d connected components' % len(connected_subG_list))

    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print('There are more than connected components. Please connect them.')
        sys.exit(1)

    return dG, leaves[0], term_size_map, term_direct_gene_map

















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










