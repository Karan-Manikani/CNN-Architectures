import os
from tensorflow.keras.datasets import mnist
import lenet

# Hide tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize x_train and x_test
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = lenet.LeNet5()
history = lenet.fit(x_train, x_test, y_train, y_test, model)
lenet.learning_curves(history)
