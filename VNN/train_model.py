import datetime
import tensorflow as tf
import numpy as np
from utils import *
#import DrugCell
#from DrugCell import *
from networks import RestrictedNN
from tensorflow.keras.utils import plot_model



# Set parameters
batch_size = 16
epochs = 100




# Generate training data.


X_train, X_test, y_train, y_test = generate_data(input_size = 1000,
                                                 input_dim = 4, 
                                                 noise = 0, 
                                                 lower = -10,
                                                 upper = 10)




X = np.array([[2, 2, 2, 2], 
              [1, 1, 1, 1]]).astype("float64")












log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



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
#print(type(my_model))
#print(my_model.inputs)
#print(my_model.outputs)

#model = tf.keras.Model(inputs=my_model.inputs, outputs=my_model.outputs)



batch_size = 1
epochs = 100



my_model.compile(loss='mean_squared_error', optimizer = "adam")

my_model.build(input_shape = (batch_size, ngene))
my_model.summary()


print("Training restricted NN")
res_history = my_model.fit(
        X_train, y_train, 
        epochs = epochs, batch_size = batch_size, 
        validation_split = 0.2,
        callbacks=[tensorboard_callback]) 





#plot_model(my_model, to_file = "functional_nn.png")
#img = plt.imread("functional_nn.png")
#plt.imshow(img)
#plt.show()
#model = drugcell_nn(term_direct_gene_map, dG, ngene, root, term_neurons)
