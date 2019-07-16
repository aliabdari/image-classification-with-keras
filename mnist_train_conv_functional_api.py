#load mnist dataset
from keras.datasets import mnist

# for creating a one hot vector for labels
from keras.utils import np_utils

#import the models
from keras import Model
#add layers
from keras import layers
#add optimizer
from keras import optimizers
#add loss function 
from keras import losses

import numpy as np

from plot_history import plot_diagrams_trian_process

#load train and test data
def prepare_data():
    (train_images,train_labels),(test_images,test_labels)=mnist.load_data()
    
    #check the shape,dimesion and type
    print("dimension = ",train_images.ndim)
    print("shape = ",train_images.shape)
    print("type = ",train_images.dtype)
    
    '''
    reshape to be ready for CNN -- 1 is the number of channel
    here image have just one channel
    '''
    X_train=train_images.reshape(60000,28,28,1)
    X_test=test_images.reshape(10000,28,28,1)
    
    #convert to float32 
    X_train=X_train.astype('float32')
    X_test=X_test.astype('float32')
    
    #normalizization
    X_train/=255
    X_test/=255
    
    #one hot enconding for labels of train and test
    Y_train = np_utils.to_categorical(train_labels)
    Y_test=np_utils.to_categorical(test_labels)
    
    return (X_train, Y_train), (X_test, Y_test)


# create the model 
def create_model():
    
    '''
    the difference between functional api and sequential:
        we should specify the inmput of every layer vice versa to the sequential model that
        we create our model and just add layers to our model
    '''
    
    #adding first layer    
    network_input = layers.Input(shape=(28,28,1))
    '''
    add a convolutional layer
    when we set the padding 'same', it means that the output size will be equal to the input size
    but if we set the padding as 'valid' no padding will be used, so the output size will be a little smaller
    '''
    
    conv1 = layers.Conv2D(filters=16,kernel_size=3,padding='same')(network_input)
    #add pooling layer
    max_pool1=layers.MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = layers.Conv2D(filters=32,kernel_size=3,padding='same')(max_pool1)
    max_pool2=layers.MaxPool2D(pool_size=(2,2))(conv2)
    #add a flatten layer to convert the image to a vector
    flat_output=layers.Flatten()(max_pool2)
    #add a classifier layer
    output=layers.Dense(10,activation='softmax')(flat_output)
    
    
    '''
    Also we can eliminate max pooling layers and using stride instead.
    By this work the speed of tarining will increase.
    conv1 = layers.Conv2D(filters=16,kernel_size=3,padding='same',strides=2)(network_input)
    conv2 = layers.Conv2D(filters=32,kernel_size=3,padding='same',strides=2)(conv1)
    flat_output=layers.Flatten()(conv2)
    output=layers.Dense(10,activation='softmax')(flat_output)
    '''
    
    '''
    now we create our model
    we should specify the input and out putlayer
    '''
    model=Model(network_input,output)
    
    #check the model structure 
    model.summary()
    
    return model

#compile will configure the model for training. optimizer and loss is essential for this function 
def compile_model(model):
    model.compile(optimizer=optimizers.Adam(),loss=losses.categorical_crossentropy,metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_label):
    #initilize epochs
    epochs_num=12
    
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

























