import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from Preprocessing_Data import scaled_test_sample as test_sample
from Preprocessing_Data import test_labels
from ANN import model


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

print("\n ---- Making predictions using the model ---- \n")
predictions = model.predict(test_sample, batch_size=10, verbose=0)

#for i in predictions:
#    print(i)

rounded_predictions = model.predict_classes(test_sample, batch_size=10, verbose=0)

#for i in rounded_predictions:
#    print(i)

print("\n ---- Confusion matrix ---- \n")

cm = confusion_matrix(test_labels, rounded_predictions)

cm_plot_labels = ['No side effects', 'Had side effects']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')