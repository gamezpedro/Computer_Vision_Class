import numpy as np
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

train_path = "./train"
test_path = "./test"

train_datagen = ImageDataGenerator(validation_split=0.2)
test_datagen = ImageDataGenerator()

train_batches = train_datagen.flow_from_directory(train_path, target_size=(224,224), classes=['dog','cat'], batch_size=10, subset='training')
validation_batches = train_datagen.flow_from_directory(train_path, target_size=(224,224), classes=['dog','cat'], batch_size=4, subset='validation') 
test_batches = test_datagen.flow_from_directory (test_path, target_size=(224,224), classes=['dog','cat'], batch_size=10) 

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

# Transform Dense into Sequential   
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))
model.summary()    

'''
    TRAINING
'''
model.compile(Adam(lr=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=4, validation_data=validation_batches,
                    validation_steps=4, epochs=5, verbose=2)

'''
    PREDICT
'''
test_imgs, test_labels = next(train_batches)
test_labels = test_labels[:,0]

predictions = model.predict_generator(test_batches, steps=1, verbose=0)

cm = confusion_matrix(test_labels, np.round(predictions[:,0]))

cm_plot_labels = ['cat','dog']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion_matrix')