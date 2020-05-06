'''

    A SINGLE HIDDEN LAYER MODEL

'''
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from Preprocessing_Data import scaled_train_sample as train_sample
from Preprocessing_Data import scaled_test_sample as test_sample
from Preprocessing_Data import train_labels, test_labels

# Keras sequential model is a linear stack of layers
# model = Sequential([l1,l2,l3]) - Option 1
# model.add(l4)                  - Option 2

print("\n  ---- Creation of the model ---- \n")

model = Sequential([
    #Dense(number_of_neurons, input_shape(for the first layer), activation_function)
    Dense(16, input_shape=(1,), activation='relu'), # Input layer
    Dense(32, activation='relu'),
    Dense(2, activation='softmax') # Output layer
])

# A quick visualization of what your model looks like
model.summary()

# Compile the model
# model.compile(optimization_function, loss_function, metrics to judge performance)
model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) #lr -> learning rate

# Validation set format
# valid_set = [(sample, label), (sample, label), ... , (sample, label)]
# or split the training data and labels when training the model

# Train the model
# model.fit(train_samples, train_labels, validation_data, batch_size, epochs, shuffle = True (default), verbose=0,1,2)
model.fit(train_sample, train_labels, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=2)

# Results showed in the training samples a slow loss and a great accuracy, but 
# the validation loss was over 1 and the accuracy couldn't get past 50%.
# the model was overfitting.