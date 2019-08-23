from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

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
    #1/(1+e^z) is our sigmoid function, utilized in scaling inputs between 0 and 1
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
    #relu: 
    model.add(Dense(units=num_classes,activation='softmax'))
    return model

##          3           ## FIX THIS
def three_layer_model_residual():
    model = Sequential()
    model.add(Dense(units=num_pixels,input_dim=num_pixels,activation='sigmoid'))
    model.add(Dense(units=num_pixels, activation='sigmoid')) ##ADD RESIDUAL LAYER!!
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


def evluate(model,batch_size=128,epochs=5):
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    history = model.fit(xtrain,ytrain,validation_split=0.1,epochs=epochs,batch_size=batch_size,verbose=False)
    loss, accuracy = model.evaluate(xtest,ytest,verbose=False)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

# model3 / model1000/ modeltanh / modelrelu / modelresidual / model5

## EDIT 'model3' BELOW TO TRY THE DIFFERENT NUERAL NETS ##
## EDIT 'batch_size' and 'epochs' AT YOUR DISCRETION 

evluate(modelrelu,batch_size=128,epochs=5)

##develop some logic behind batch size and epoch and using adam vs sgd
##analyze all graphs from each model and talk about why they did what they did!
##residual layer ann

##GOOD JOB MRIDUL! :)