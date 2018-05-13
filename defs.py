# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 02:00:23 2018

@author: kira
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from numpy.random import permutation

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers import Conv2D,Conv1D, MaxPooling2D, MaxPooling1D
from keras.optimizers import SGD, RMSprop

import itertools
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import argparse
import _pickle
import matplotlib.pyplot as plt
from keras.models import load_model
import time

maxlen = 76


def create_model(dense_layers=[16,8,4],
                 activation=['relu','relu','relu'],
                 optimizer='rmsprop'):
    model = Sequential()
    model.add(Conv1D(64,3, input_shape=(None,maxlen), padding='same'))
#mirs_branch.add(Dense(64))
    model.add(Activation(activation[0]))
    model.add(MaxPooling1D(pool_size=3,padding='same'))
    #model.add(Dropout(0.25))

    for index, lsize in enumerate(dense_layers):
        # Input Layer - includes the input_shape
        #if index == 0:
            #model.add(Dense(lsize,
                           # activation=activation,
                            #input_shape=(4,)))
        #else:
            model.add(Dense(lsize,
                            activation=activation[1]))
            model.add(Dropout(0.25))
            
    model.add(Dense(1,activation=activation[2]))
    model.compile(optimizer = optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

def search_params(diry,x_train,y_train,x_test,y_test):
    print("x_train,y_train shape: ",x_train.shape,y_train.shape)
    #x_train = x_train.reshape(x_train.shape[0],-1)    
    #y_train = y_train.reshape(y_train.shape[0],1)  
    accuracies = []
    epochs = 300
    with open(diry+'search_params.txt','w') as f:#, open(diry+'search_models.txt','w') as f2:
        
        for den in [[64,32,8]]:
        #[[32,16,8,8],[32,32,16,16,8,4]]:
        #[[64,32,32,16],[64,32,16],[64,32,8],[64,16,16,4],[64,32,16,8,8]]: 
        #[[32,8,4],[32,8],[8,8],[16,4]]:
            for act in [['relu','relu','relu']]:
            #[['relu','relu','relu']]:#,['relu','relu','sigmoid'],['tanh','tanh','relu']]:
                sgd_lr0001_dec1e6_m095= SGD(lr=0.001, decay=1e-6, momentum=0.95, nesterov=True)
                for opt in ['adam']:# [sgd_lr0001_dec1e6_m09,'rmsprop','adam']:
                    start_time = time.time()
                    model = create_model(den,act,opt)
                    # Pass the file handle in as a lambda function to make it callable
                    model.summary(print_fn=lambda x: f.write(x + '\n'))
                    history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=50,
                    verbose = 2,
                    validation_split=0.3)
                    #train_time = time.time()
                    
                    if opt==sgd_lr0001_dec1e6_m095:
                        opt='sgd_lr0001_dec1e6_m095'
                    f.write('\ndense layers: %s\n activations: %s\n optimizer: %s '%(str(den),str(act),str(opt)))
                    f.write('\nTraining duration (s) : '+ str(time.time() - start_time))                    
                    f.write("\n Final validation accuracy: \n")
                    f.write(str(history.history['val_acc'][-1]))
                    # summarize history for accuracy
                    plt.plot(history.history['acc'])
                    plt.plot(history.history['val_acc'])
                    plt.title('dense layers: %s\n activations: %s\n optimizer: %s \n accuracy'%(str(den),str(act),str(opt)))
                    plt.ylabel('accuracy')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'validation'])
                    plt.savefig(diry+'accuracy_%s%s%s.png'%(str(den),str(act),str(opt)))
                    #plt.show()
                    plt.clf()
# summarize history for loss
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('dense layers: %s\n activations: %s\n optimizer: %s \n loss'%(str(den),str(act),str(opt)))
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'validation'])
                    plt.savefig(diry+'loss_%s%s%s.png'%(str(den),str(act),str(opt)))
                    #plt.show()
                    plt.clf()
                    test_start_time = time.time()
                    y_pred = model.predict(x_test)
                    y_pred = (y_pred>0.5)
                    y_true=y_test.reshape(y_test.shape[0])
                    y_pred=y_pred.reshape(y_pred.shape[0])
    # 3. Print accuracy score
                    f.write("\n *************************************************** \n \n")
                    acc_score = accuracy_score(y_true,y_pred)
                    accuracies.append(acc_score)
                    f.write("Test accuracy : "+ str(acc_score))
                    f.write('\nTesting duration (s) : '+ str(time.time() - test_start_time))

                    f.write("\nClassification Report\n")
                    f.write(classification_report(y_true,y_pred,digits=5))    
    # 5. Plot confusion matrix
                    f.write("\nConfusion_matrix\n")
                    f.write("true| predicted label\n")
                    f.write("    |  0    1\n")
                    cnf_matrix = confusion_matrix(y_true,y_pred)
                    f.write('0   '+str(cnf_matrix[0]))
                    f.write('\n1   '+str(cnf_matrix[1]))
                    f.write("\n *************************************************** \n \n")
        print("\n Final test accuracies: ",accuracies)
        f.write("\n Final test accuracies: "+str(accuracies))
    f.close()
    
    #return model



def train_model(diry,alin, y_target):
    # build the neural net architecture
    model = Sequential()
    model.add(Conv1D(64,3, input_shape=(None,alin.shape[2]), padding='same'))
#mirs_branch.add(Dense(64))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3,padding='same'))
    model.add(Dropout(0.25))
    #model.add(Dense(64))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8))
    model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(2))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('relu')) 

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='mean_squared_error', #'binary_crossentropy',
              optimizer='adam', #sgd,
              metrics=['accuracy'])
# train the model
    history = model.fit(alin, y_target,
              batch_size=50,
              epochs=70,
              validation_split=0.3)
    with open(diry+'history.pickle', 'wb') as f:
        _pickle.dump(history.history, f)
    with open(diry+'history.csv', 'wb') as f:
        _pickle.dump(history.history, f)
    # list all data in history
    print(history.history.keys())
# summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(diry+'accuracy_plot.png')
    plt.show()
    plt.clf()
# summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(diry+'loss_plot.png')
    plt.show()
    plt.clf()
    model.save(diry+'conv64-32-8-1.h5')
    return model
# define a function for data preprocessing

    
## multiclass or binary report
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(diry,model, x, y_true, classes, binary=True,normalize=True):
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict(x)
    y_pred = (y_pred>0.5)
    y_true=y_true.reshape(y_true.shape[0])
    y_pred=y_pred.reshape(y_pred.shape[0])
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    print("")
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    #plot_confusion_matrix(cnf_matrix,classes=classes)
    cm = cnf_matrix
    cmap=plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(diry+title+'.png')
    plt.show()
    
def traintest_split(alin,y_target):
    test_len = int(y_target.shape[0]*0.2)
    y_target_test = y_target[-test_len:]
    alin_test = alin[-test_len:]
    y_target = y_target[:-test_len]
    alin = alin[:-test_len]
    return alin,y_target,alin_test,y_target_test