#.\venv\Scripts\activate

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as tlp

X,y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

xtrain, xtest, ytrain, ytest = X[:60000], X[60000:], y[:60000], y[60000:]

#print(xtrain.shape) #(60000, 28,28)
#print(xtest.shape) #(10000, 28, 28)

#firstdigit = xtrain[0] 
#firstdigitimage = firstdigit.reshape(28,28) 

#tlp.imshow(firstdigitimage, cmap = mpl.cm.binary, interpolation = 'nearest')
#tlp.axis('off')
#tlp.show() # first number is 5... lets utilize this to create a binary classifier for the number 5
image_vector = 28*28 # 784

xtrain = xtrain.reshape(xtrain.shape[0],image_vector) #  60000x784 matrix
xtest = xtest.reshape(xtest.shape[0],image_vector) #10000x784 matrix

###                 binary classifier                   ###


ytrain_5 = (ytrain == 5) #sets all 5's in training set to 1 everything else set to 0

ytest_5 = (ytest == 5) #sets all 5's in test set to 1 everything else set to 0

#   Utilize Stoachastic Gradient Descent   #

from sklearn import linear_model

sgdclf = linear_model.SGDClassifier(alpha = 1, epsilon = 0.1, learning_rate='optimal', max_iter = 100, random_state = 42) #random_state assures replicable state on other machines
sgdclf.fit(xtrain,ytrain_5)

#   Accuracy via CV dataset   #

from sklearn import model_selection

print(model_selection.cross_val_score(sgdclf,xtrain,ytrain_5, cv = 5,scoring='accuracy')) 
##though accuracy does not tell us anything, we should analyze percision,recall and in turn calculate the F1 score
ytrain_pred = model_selection.cross_val_predict(sgdclf,xtrain,ytrain_5,cv = 5)

from sklearn import metrics
precision = metrics.precision_score(ytrain_5, ytrain_pred)
recall = metrics.recall_score(ytrain_5,ytrain_pred)
F1score = metrics.f1_score(ytrain_5, ytrain_pred)
print('percision:\n',precision,'\nrecall:\n', recall,'\nF1score:\n',F1score)