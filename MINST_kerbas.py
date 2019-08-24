from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization, ReLU, Add, Input, Flatten, merge
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import keras
import random

#   Load MNIST data   #

(xtrain,ytrain), (xtest,ytest) = mnist.load_data()

# xtrain and ytrain shape (60000, 28, 28) 
# xtest and ytest shape (10000, 28,28)

#   Vectorize image data   #

num_pixels = xtrain.shape[1] * xtrain.shape[2]
# num_pixels = 784
xtrain = xtrain.reshape(xtrain.shape[0], num_pixels).astype('float32')
# xtrain.shape = (60000, 784)
xtest = xtest.reshape(xtest.shape[0], num_pixels).astype('float32')
#xtest.shape = (60000,1)

#   Normalize Features   #

xtrain = xtrain/255 
xtest = xtest/255

#   Transforming vector of classes to vector of binary values   #

ytrain = np_utils.to_categorical(ytrain)
#ytrain.shape (60000,10)
ytest = np_utils.to_categorical(ytest)
#ytest.shape (10000,10)
num_classes = ytest.shape[1]
#num_classes = 10

#   Nueral Networks   #


##                      Numbers correspond to slack message from Justin                 #


##          0           ##
def three_layer_model():
    model = Sequential()
    # Hidden Layer 1 #
    model.add(Dense(units=num_pixels,input_dim=num_pixels,activation='sigmoid'))
    #num_pixels = 784
    # Hidden Layer 2 #
    model.add(Dense(units=num_pixels, activation='sigmoid'))
    #1/(1+e^z) is our sigmoid function, utilized in scaling between 0 and 1 has disadvantage of vanishing gradients 
    # Output Layer #
    model.add(Dense(units=num_classes,activation='softmax'))
    #softmax scales values between 0 and 1 to give probabilites of which category input image could be
    return model

##          1           ##
def three_layer_model_1000():
    model = Sequential()
    model.add(Dense(units=num_pixels,input_dim=num_pixels,activation='sigmoid'))
    model.add(Dense(units=1000, activation='sigmoid'))
    #1000x1000 layer
    model.add(Dense(units=num_classes,activation='softmax'))
    return model

##          2           ##
def three_layer_model_tanh():
    model = Sequential()
    model.add(Dense(units=num_pixels,input_dim=num_pixels,activation='tanh'))
    model.add(Dense(units=num_pixels, activation='tanh'))
    model.add(Dense(units=num_classes,activation='softmax'))
    return model
def three_layer_model_relu():
    model = Sequential()
    model.add(Dense(units=num_pixels,input_dim=num_pixels,activation='relu'))
    model.add(Dense(units=num_pixels, activation='relu'))
    #relu: aids us with our vanishing gradient problem, leaky relu would solve it entirely
    model.add(Dense(units=num_classes,activation='softmax'))
    return model

##          3           ## 

def three_layer_model_residual():
    prob=random.uniform(0, 1)
    threshold=0.4
    model = Sequential()
    model.add(Dense(units=num_pixels,input_dim=num_pixels,activation='relu'))
    #If prob is greater than threshold then a layer in the middle will be added, otherwise it will be a 2 layer network... further steps could include to optimize the threshold for best results
    if prob >= threshold:
        model.add(Dense(units=num_pixels,input_dim=num_pixels, activation='relu'))
    model.add(Dense(units=num_classes,activation='softmax'))
    return model

##          4           ##
def five_layer_model():
    model = Sequential()
    # Hidden Layers 1,2,3,4 #
    model.add(Dense(num_pixels, input_dim=num_pixels,activation='sigmoid'))
    model.add(Dense(num_pixels,activation='sigmoid'))
    model.add(Dense(num_pixels,activation='sigmoid'))
    model.add(Dense(num_pixels,activation='sigmoid'))
    # Output Layer #
    model.add(Dense(units=num_classes,activation='softmax'))
    return model

model3=three_layer_model()
model1000=three_layer_model_1000()
modeltanh=three_layer_model_tanh()
modelrelu=three_layer_model_relu()
modelresidual=three_layer_model_residual()
model5=five_layer_model()

i=1 ##tracks which model is being used


def evaluate(model,batch_size=128,epochs=5):
    #Models accuracy and error via Keras
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    history = model.fit(xtrain,ytrain,validation_split=0.1,epochs=epochs,batch_size=batch_size,verbose=False)
    loss, accuracy = model.evaluate(xtest,ytest,verbose=False)
    #categorical_crossentropy: utlized as our categories are one-hot encoded, best optimization score function for our situation
    #sgd vs adam: sgd gave lower accuracy in 5 layer model (which is expected as it should be overfitted) but adam gave same result even on 5 layer, facisinating...\
    

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel(print('epochs\n',epochs),print('\nbatch size\n',batch_size),print('\nmodel\n',i))
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')


## EDIT 'model3' BELOW TO TRY THE DIFFERENT NUERAL NETS ##
# model3 / model1000/ modeltanh / modelrelu / modelresidual / model5

## EDIT 'batch_size' and 'epochs' AT YOUR DISCRETION ##
#batch_size: how many datapoints are fed to the ANN
#epochs: number of times ANN will be trained

evaluate(model=model3,batch_size=64,epochs=3)

#models = [model1000,modeltanh,modelrelu,modelresidual,model5]
#batches = [32,64,128]
#epochs = [3,5,7]

#for m in models:
#    for b,e in zip(batches,epochs):
#       evaluate(model=m,batch_size=b,epochs=e)
#i=i+1
