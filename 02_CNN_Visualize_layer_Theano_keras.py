'''
Visulize CNN Layer output in Keras
1) Extracting weights of each layers and also getting total parameters of the entire model
2) Visualizing filters
3) Getting layer configuration
4) Visualizing intermediate layer output
'''
#%%

# -*- coding: utf-8 -*-
#print("welcome")
#%%
from keras.datasets import mnist # import dataset
from keras.models import Sequential # import type of model
from keras.layers.core import Dense, Dropout, Activation, Flatten  # import layes
from keras.layers.convolutional import Convolution2D, MaxPooling2D #import conolution layes
from keras.utils import np_utils
import theano
import numpy as np
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
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_class))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=["accuracy"])
#%%
model.fit(x_train,y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_data=(x_test,y_test))

#%%
score=model.evaluate(x_test,y_test,verbose=0)
print('Test_score:',score[0])
print('Test_accuracy:',score[1])
print(model.predict_classes(x_test[1:5]))
print(y_test[1:5])
#%%
"""
model.get_config() # gets the configuration of convolution layer
model.layers[0].get_config() # get the configuration of first layer layer 0
model.layers[0].count_params()

"""


def plot_filters(layer,x,y):
    """ Plot the filters for net after the convolutional layer. They are trained in x and y Format
	so for example if we haven20nfilters after layer 0 , then we can call plot_filters(1_conv1,5,4) tobytesget 5 by 4 plot of all filt
	ers."""
    filters =layer.W.get_value()
    fig = plt.figure()
    for j in range(len(filters)):
        ax =fig.add_subplot(y,x,j+1)
        ax.matshow(filters[j][0],cmap =matplotlib.cm.binary) # for conv1-filter shape [32,1,,5,5]
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return (plt)
		
		
# plot the first convolutional layer filters
plot_filters(model.layers[0],8,4)
#%%

#visulaizing intermittent layer filter

output_layer = model.layer[1].get_output()
output_fn = theano.function([model.layers[0].get_input()],output_layer)

# Input image
in =output_fn(x_train[0:1,:,:,:])
print(in.shape)

#Rearrange dimension so we can plot the result in RGB image
in=np.rollaxis(np.rollaxix(in,,1),3,1)
print(im.shape)

fig =plt.figure(figuresize=(16,8)
for i in range(32):
    plt.subplot(1,10,i+1)
	plt.imshow(in[0,:,:,i],interpolation ='nearest') # to see the first filter
	plt.axis('off')
	
fig = plt.figure(figsize=(8,8))
for i in range(32):
    ax -fig.add_subplot(6,6,i+1)
	plt.imshow(in[0,:,:,i],interpolation ='nearest') # to see the first filter
	plt.xtricks(np.array([]))
	plt.ytricks(np.array([]))
	plt.tight_layout()
plt
	




 











