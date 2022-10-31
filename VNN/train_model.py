import tensorflow as tf
from utils import *
#import DrugCell
#from DrugCell import *
from networks import RestrictedNN
from tensorflow.keras.utils import plot_model






# load ontology
gene2id_mapping = load_mapping("Data/geneID_test.txt")
dG, root, term_size_map, term_direct_gene_map = load_ontology("Data/onto2_test.txt", gene2id_mapping)

# Set the number of neurons for each term.
term_neurons = 1
ngene = 4

my_model = RestrictedNN(root=root, 
                        dG=dG, 
                        module_neurons=term_neurons, 
                        n_inp=ngene, 
                        term_direct_gene_map=term_direct_gene_map,
                        mod_size_map=term_size_map)


print(my_model)
print(type(my_model))
print(my_model.inputs)
print(my_model.outputs)

model = tf.keras.Model(inputs=my_model.inputs, outputs=my_model.outputs)

model.compile(loss='mean_squared_error', optimizer = "adam")
model.summary()

raise

raise

#plot_model(my_model, to_file = "functional_nn.png")
#img = plt.imread("functional_nn.png")
#plt.imshow(img)
#plt.show()
#model = drugcell_nn(term_direct_gene_map, dG, ngene, root, term_neurons)
