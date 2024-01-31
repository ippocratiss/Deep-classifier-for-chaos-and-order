import numpy as np
from matplotlib import pyplot
import tensorflow as tf
import keras 
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPool2D , Flatten, Activation
from generate_data import *

################################
# This module defines i) the deep model, ii) the plotting and iii) testing/evaluation functions.
################################

# Define the model
def produce_model(iterator_train, iterator_val, epochs = 20, do = 0.2, n1 = 64, n2 = 128, res = 30, learning_rate=0.006 ):
    # Initiate the sequential model
    model = Sequential()
    model.add(keras.Input(shape=(res,res,1)) )
    # Convolutional layer 1
    model.add(Conv2D(filters= n1, kernel_size= (3,3), strides= (1,1), padding='same', activation= 'relu'))
    model.add(Conv2D(filters= n1, kernel_size= (3,3), strides= (1,1), padding='same', activation= 'relu'))
    model.add(MaxPool2D(pool_size= (2,2), strides=(2,2)))
    # Convolutional layer 2
    model.add(Conv2D(filters= n2, kernel_size= (3,3), strides= (1,1), padding='same', activation= 'relu'))
    model.add(Conv2D(filters= n2, kernel_size= (3,3), strides= (1,1), padding='same', activation= 'relu'))
    model.add(MaxPool2D(pool_size= (2,2), strides=(2,2)))
    # Flatten output to prepare for dense layers input
    model.add(Flatten())
    # Dense layer
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(do))
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(do))
    # Batch normalisation + Output layer
    model.add(BatchNormalization())
    model.add(Dense(units= 2, activation='softmax'))
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate= 0.006),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
                 )
    # Fit the model and test against validation data
    history = model.fit(iterator_train, validation_data = iterator_val, epochs=epochs)
    
    return model, history 

    
# Returns the evaluation of the data set given a trained model and a data iterator.
def produce_evaluation(iterator_set, model):
# load the desired dataset as usual, 'load_XXXX'
# choose which iterator/data-generator to use (train, test, etc.)
# select which trained ML model to use for evaluation 
    iterator = iterator_set 
    batchX, batchy = iterator.next()
    print(batchX.shape, batchX.mean(), batchX.std())
    _, acc_test = model.evaluate(iterator)
    print('Test Accuracy =' , acc_test)
    return acc_test


# Runs a loop of training and evaluation to produce the mean and std. It can also save best models.
# If new_iterators = True, then at each iteration new data iterators can be defined so that a new data # augmentation is performed.
def loop_train_eval(i, data_train, data_val, iterator_train, iterator_val, iterator_test, epochs = 40, new_iterators = True):
    # By default, we use data_train, data_val as defined in the Jupyter notebook
    # Define new iterators for training/validation at each iteration if new_iterators = True
    # Testing iterator is kept fixed and defined in the Jupyter notebook. 
    if new_iterators == True:
        iterator_train =  produce_iterator(data_train, batch_size=64, print_stats=True,
                  shuffle=True, rotation_range=60, 
                  width_shift_range=0.001, height_shift_range=0.001, 
                  shear_range=0.35, fill_mode='nearest')
        iterator_val =  produce_iterator(data_val, batch_size=64, print_stats=True,
                  shuffle=True, rotation_range=60, 
                  width_shift_range=0.001, height_shift_range=0.001, 
                  shear_range=0.35,fill_mode='nearest')        
    # run the fitting function to produce the train model
    model_i, history_i = produce_model(iterator_train, iterator_val, epochs = epochs)
    # evaluate the mode3
    acc_i = produce_evaluation(iterator_test, model_i)
    if acc_i >= 0.9: # save the keras model if accuracy is greater than 90 %
        model_i.save('best_model_acc'+ str(np.round(acc_i*100)) +'.keras')        
    return acc_i  # returns the accuracy for the test model


# Plots the training/validation accuracy and loss function as a function of the epochs.
def plot_history(history):    
    import matplotlib.pyplot as plt      
    plt.plot(history.history['accuracy']) # plot training accuracy
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['loss'])  # plot loss 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.plot(history.history['val_accuracy']) # plot validation accuracy 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train_accuracy', 'Loss', 'Val_accuracy_1'], loc='upper left')
    plt.show()
