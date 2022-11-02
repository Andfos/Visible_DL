import sys 
import datetime
import tensorflow as tf
import numpy as np
from utils import *
#import DrugCell
#from DrugCell import *
from networks import RestrictedNN
from tensorflow.keras.utils import plot_model
from packaging import version
import tensorboard

# Set parameters
batch_size = 16
epochs = 100







# Set the functions that will generate the training data.
func = "x[0] * x[1] + x[2]"
noise_sd = "sqrt(abs(x[1] * x[2]))"
function_name = f"f(X) = {func} + E~N(0, {noise_sd}"

# Generate X and y vectors.
X, y = generate_data(
        function = func,
        noise_sd_func = noise_sd,
        data_size = 1000,
        input_dim = 3,  
        lower = -10,
        upper = 10)

# Split the training and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 42)



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



# load ontology
gene2id_mapping = load_mapping("Data/geneID_test.txt")
dG, root, term_size_map, term_direct_gene_map = load_ontology("Data/onto_test.txt", gene2id_mapping)

# Set the number of neurons for each term.
term_neurons = 1
ngene = 3

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

#my_model.build(input_shape = (batch_size, ngene))
#my_model.summary()





# Plot the models





print("Training restricted NN")
res_history = my_model.fit(
        X_train, y_train, 
        epochs = epochs, batch_size = batch_size, 
        validation_split = 0.2,
        callbacks=[tensorboard_callback]) 




plot_model(my_model, to_file = "functional_nn.png")
img = plt.imread("functional_nn.png")
plt.imshow(img)
plt.show()

#plot_model(my_model, to_file = "functional_nn.png")
#img = plt.imread("functional_nn.png")
#plt.imshow(img)
#plt.show()
#model = drugcell_nn(term_direct_gene_map, dG, ngene, root, term_neurons)
