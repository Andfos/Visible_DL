from utils import *
#import DrugCell
#from DrugCell import *
from networks import RestrictedNN







# load ontology
gene2id_mapping = load_mapping("geneID_test.txt")
dG, root, term_size_map, term_direct_gene_map = load_ontology("ontology_test.txt", gene2id_mapping)

# Set the number of neurons for each term.
term_neurons = 1
ngene = 3

my_model = RestrictedNN(root=root, 
                        dG=dG, 
                        module_neurons=term_neurons, 
                        n_inp=ngene, 
                        term_direct_gene_map=term_direct_gene_map,
                        term_size_map=term_size_map)

#model = drugcell_nn(term_direct_gene_map, dG, ngene, root, term_neurons)
