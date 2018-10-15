import itertools
import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
import sys
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if not len(sys.argv) == 3:
    sys.exit("Usage: myOwnKeras training1 training2 test1 test2")

test1 = sys.argv[1]
test2 = sys.argv[2]

print(test2, test1)

data_test1 = np.genfromtxt(test1, delimiter=',')
data_test2 = np.genfromtxt(test2, delimiter=',')


data_test1_y = np.zeros((data_test1.shape[0], 1))
data_test2_y = np.ones((data_test2.shape[0], 1))


test_x = np.concatenate((data_test1, data_test2), axis=0)
#test_y = np.concatenate((data_test1_y, data_test2_y), axis=0)


test_y = np_utils.to_categorical(np.concatenate((data_test1_y, data_test2_y), axis=0), 2)


print("-> Data samples loaded!")
print("-> Shuffling samples...")
test_x, test_y = shuffle(test_x, test_y)

print("Done!")

#start_time = time.time()


print("->Loading model...")
model = load_model('Models/50epochs_50batch.h5')
print("Loaded!")



print("-> Evaluating the model...")
score = model.evaluate(test_x, test_y, batch_size=1)
print("Network's test score [loss, accuracy]: {0}".format(score)+"\n")


print("-> Predicting the samples and computing confusion matrix...")

pred_y = model.predict(test_x, batch_size=1, verbose = 1)
pred_y = np.argmax(pred_y, axis = 1)
test_y = np.argmax(test_y, axis = 1)
print(pred_y)
print(test_y)


cnf_matrix = confusion_matrix(test_y, pred_y)

class_names = ["DY", "ttbar"]

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig("Results/ConfusionMatrix.png")

print("Confusion matrix created in Results/ConfusionMatrix.png")


#print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
#print("Network's test score [loss, accuracy]: {0}".format(score))


#model.save("Models/"+str(n_epochs)+"epochs_"+str(n_batch)+"batch.h5")





