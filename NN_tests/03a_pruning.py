from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
from sklearn.model_selection import train_test_split
import tempfile


validation_split = 0.2
batch_size = 16
epochs = 2000
upper = 5
lower = -5
noise = 0
input_dim = 1
input_size = 1000









def generate_data(input_size, input_dim, noise):
    """ Create the Input and output """

    X = np.zeros(shape = (input_size, input_dim))
    y = np.zeros(shape = (input_size, 1))

    for i in range(0, input_size):
        #input_list = np.random.randint(low = lower, high = upper, size = input_dim)
        input_list = [np.random.uniform(lower, upper)]
        output = input_list[0] ** 2 #+ np.random.normal(0, noise, 1)

        X[i] = input_list
        y[i] = output

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2,
                                                        random_state = 42)

    return X_train, X_test, y_train, y_test




X_train, X_test, y_train, y_test = generate_data(
        input_size, 
        input_dim,
        noise)



model = keras.models.load_model("Trained_Models/prunedOnce_2Lin_2Sig")
#model = keras.models.load_model("Trained_Models/Dense_2Lin_2Sig")


init_weights = model.get_weights()





prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
          #initial_sparsity = 0,
          #final_sparsity = .25, 
          target_sparsity = 0.25,
          begin_step = 0, 
          end_step = -1)}


model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer = "adam", loss = "mse")

model_for_pruning.summary()



logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]




model_for_pruning.fit(X_train, y_train,
                  batch_size=batch_size, epochs=epochs,
                      validation_split=validation_split, 
                      callbacks = callbacks)


pruned_weights = model_for_pruning.get_weights()


print("Initial weights:\n{}\n\n".format(init_weights))
print("Pruned weights:\n{}\n\n".format(pruned_weights))






new_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
new_weights = new_model.get_weights()
print(new_weights)



#new_model.save("Trained_Models/prunedOnce_2Lin_2Sig")



