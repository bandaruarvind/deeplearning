# -*- coding: utf-8 -*-
#print("welcome")
#%%
from keras.datasets import mnist # import dataset
from keras.models import Sequential # import type of model
from keras.layers.core import Dense, Dropout, Activation, Flatten  # import layes
from keras.layers.convolutional import Convolution2D, MaxPooling2D #import conolution layes
from keras.utils import np_utils

# To plot import matplotlib

import matplotlib
import matplotlib.pyplot as plt

#%%

#batch size to train
batch_size =128
# Number of out put class 10
nb_class =10
#Number of epochs
nb_epoch=1
#input image dimension
img_row, img_col =28,28
#number of convolution filters
nb_filter =32
#size of pooling area for maxpooling
nb_pool =2
#convolution keranl size
nb_conv=3
#%%
#data to be shuffed and split between train and test
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#reshape the data
x_train=x_train.reshape(x_train.shape[0],1,img_row,img_col)
x_test=x_test.reshape(x_test.shape[0],1,img_row,img_col)

#convert to dtype unit to float
x_train=x_train.astype("float32")
x_test=x_test.astype("float32")

#normalize the data 0-255 by dividing with 255
x_train /= 255
x_test /=255

print("x_train shape:", x_train.shape)
print(x_train.shape[0],"train samples")
print(x_test.shape[0],"test samples")

#%%
#convert class vector to one-hot encloding for cross entropy loss
y_train =np_utils.to_categorical(y_train,nb_class)
y_test =np_utils.to_categorical(y_test,nb_class)

#%%
i=600
plt.imshow(x_train[i,0],interpolation='nearest')
print("label:",y_train[i,:])

#%%
#defining model now

model =Sequential()
model.add(Convolution2D(nb_filter,nb_conv,nb_conv,border_mode='valid',input_shape=(1,img_row,img_col)))
convout1=Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filter,nb_conv,nb_conv))
convout2=Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_class))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta')
#%%
model.fit(x_train,y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_data=(x_test,y_test))

#%%
score=model.evaluate(x_test,y_test,verbose=0)
print('Test_score:',score[0])
print('Test_accuracy:',score[1])
print(model.predict_classes(x_test[1:5]))
print(y_test[1:5])




