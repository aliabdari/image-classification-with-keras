#import pyplot for plotting accuracy and loss trend in training step
import matplotlib.pyplot as plt

def plot_diagrams_trian_process(history):
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['loss','val_loss'])
    
    plt.figure()
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.legend(['acc','val_acc'])