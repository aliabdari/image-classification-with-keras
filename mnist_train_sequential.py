#load mnist dataset
from keras.datasets import mnist

# for creating a one hot vector for labels
from keras.utils import np_utils

#import the model. here we using keras sequential model
from keras.models import Sequential
#add fully connected and dropout layer 
from keras.layers import Dense,Dropout
#add SGD optimizer
from keras.optimizers import SGD
#add loss function 
from keras.losses import categorical_crossentropy

import numpy as np

from plot_history import plot_diagrams_trian_process

#load train and test data
def prepare_data():
    (train_images,train_labels),(test_images,test_labels)=mnist.load_data()
    
    #check the shape,dimesion and type
    print("dimension = ",train_images.ndim)
    print("shape = ",train_images.shape)
    print("type = ",train_images.dtype)
    
    #reshape to be ready for NN
    X_train=train_images.reshape(60000,784)
    X_test=test_images.reshape(10000,784)
    
    #convert to float32 
    X_train=X_train.astype('float32')
    X_test=X_test.astype('float32')
    
    #normalize
    X_train/=255
    X_test/=255
    
    #one hot enconding for labels of train and test
    Y_train = np_utils.to_categorical(train_labels)
    Y_test=np_utils.to_categorical(test_labels)
    
    return (X_train, Y_train), (X_test, Y_test)


# create the model 
def create_model():
    model=Sequential()
    #add some fully connected layers
    model.add(Dense(500,activation='relu',input_shape=(784,)))
    #add dropout layer     
    model.add(Dropout(20))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(20))
    model.add(Dense(10,activation='softmax'))
    
    #check the model structure 
    model.summary()
    
    return model

#compile will configure the model for training. optimizer and loss is essential for this function 
def compile_model(model):
    model.compile(optimizer=SGD(lr=0.001),loss=categorical_crossentropy,metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_label):
    #initilize epochs
    epochs_num=50
    
    #initialize batch size
    batch_size_num=100
    
    #train the model -- assign 20% of train data for validation purpose 
    network_history = model.fit(X_train,Y_label,batch_size=batch_size_num,epochs=epochs_num,validation_split=0.2)
    
    return network_history.history


def evaluate(model,X,Y):
    #evaluate the model
    test_loss,test_acc = model.evaluate(X,Y)
    print('test accuracy = %f and test loss = %f'%(test_acc,test_loss))
    #predict the new inputs
    predicted_outpout = model.predict(X)
    predicted_outpout=np.argmax(predicted_outpout,axis=1)

    
def main():
    (X_train, Y_train),(X_test, Y_test) = prepare_data()
    model = create_model()
    model = compile_model(model)
    history = train_model(model, X_train, Y_train)
    plot_diagrams_trian_process(history)
    evaluate(model,X_test,Y_test)

if __name__ == '__main__':
    main()

























