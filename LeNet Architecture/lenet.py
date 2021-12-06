# Packages
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Hide tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def LeNet5():
    """
    Implementation of LeNet - 5 CNN architecture as in the original paper using the keras Functional API
    ZeroPadding -> Conv2D -> AveragePooling -> Conv2D -> AveragePooling -> FC layer -> FC layer -> Softmax 

    Returns:
        leNet -- keras.Model() instance
    """

    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.ZeroPadding2D(padding=(2, 2))(inputs)
    x = layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='tanh')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='tanh')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=120, activation='relu')(x)
    x = layers.Dense(units=84, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)
    leNet = keras.Model(inputs=inputs, outputs=outputs)
    leNet.summary()

    return leNet


def fit(x_train, x_test, y_train, y_test, model, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=None, batch_size=32, epochs=10):
    """
    Inputs:
        x_train -- training set features 
        x_test -- testing set features 
        y_train -- training set targets 
        y_test -- testing set targets 
        model -- keras.Model() instance 
        loss -- cost function to be optimized
        metrics -- metrics used to evaluate model 
        batch_size -- size of minibatch 
        epochs -- number of epochs 

    Outputs:
        history -- keras.callbacks.History instance
    """

    if metrics is None:
        metrics = ['accuracy']

    model.compile(
        loss=loss,
        metrics=metrics
    )

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    model.evaluate(x_test, y_test, batch_size=batch_size)

    return history


def learning_curves(history, epochs=10):
    """
    Plots the loss and accuracy curves for the training dataset
    
    Inputs:
        history -- keras.callbacks.History instance 
        epochs -- number of epochs 
    
    Outputs:
        learning curves 
    """

    loss = history.history['loss']
    accuracy = history.history['accuracy']
    epochs = range(1, epochs + 1)

    plt.plot(epochs, loss, 'r', label='Loss')
    plt.plot(epochs, accuracy, 'g', label='Accuracy')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
