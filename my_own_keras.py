import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
import sys


class LossHistory(cb.Callback):
   
    # Class to store the training information: loss and accuracy
 
    def on_train_begin(self, logs={}):
	
      	# Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    def on_epoch_end(self, batch, logs={}):
        
        # This function is called at the end of each epoch

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        
def plot_history(history):

    """
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    """
    
    loss_list = history.losses
    acc_list = history.acc
    val_loss_list = history.val_losses
    val_acc_list = history.val_acc

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(loss_list) + 1)
   

    plt.figure(1, figsize = (6.4, 9.6)) 
    ## Loss
    plt.subplot(211)
    plt.plot(epochs, loss_list, 'b', label='Training loss')
    plt.plot(epochs, val_loss_list, 'g', label = 'Validation loss')
    """
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    """
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.subplot(212)
    plt.plot(epochs, acc_list, "b", label = 'Training accuracy')
    plt.plot(epochs, val_acc_list, 'g', label = 'Validation accuracy')
    """
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    """
    
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig("History.png")



if not len(sys.argv) == 5:
    sys.exit("Usage: myOwnKeras training1 training2 test1 test2")


# -> Get the training and test files to train the net
training0 = sys.argv[1]
training1 = sys.argv[2]
test0 = sys.argv[3]
test1 = sys.argv[4]

print("-> The files used are: \n")
print("Class 0: "+training0 +" "+test0)
print("Class 1: "+training1 +" "+test1+"\n")


# -> Create the samples from the test
data_training0 = np.genfromtxt(training0, delimiter=',')
data_training1 = np.genfromtxt(training1, delimiter=',')
data_test0 = np.genfromtxt(test0, delimiter=',')
data_test1 = np.genfromtxt(test1, delimiter=',')

data_training0_y = np.zeros((data_training0.shape[0], 1))
data_training1_y = np.ones((data_training1.shape[0], 1))
data_test0_y = np.zeros((data_test0.shape[0], 1))
data_test1_y = np.ones((data_test1.shape[0], 1))

training_x = np.concatenate((data_training0, data_training1), axis=0)
training_y = np_utils.to_categorical(np.concatenate((data_training0_y, data_training1_y), axis=0), 2)
test_x = np.concatenate((data_test0, data_test1), axis=0)
test_y = np_utils.to_categorical(np.concatenate((data_test0_y, data_test1_y), axis=0), 2)

start_time = time.time()
print 'Compiling Model ... '
model = Sequential()
model.add(Dense(24, input_dim=12))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('sigmoid'))
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
print 'Model compield in {0} seconds'.format(time.time() - start_time)


history = LossHistory()
print 'Training model...'

n_epochs = 50
n_batch = 50
model.fit(training_x, training_y, epochs=n_epochs, batch_size=n_batch,
          callbacks=[history],
          validation_data=(test_x, test_y), verbose=1, shuffle = True)

plot_history(history)

print "Training duration : {0}".format(time.time() - start_time)
score = model.evaluate(test_x, test_y, batch_size=1) 
print "Network's test score [loss, accuracy]: {0}".format(score)


model.save("Models/"+str(n_epochs)+"epochs_"+str(n_batch)+"batch.h5")





