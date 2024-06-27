import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
mnist = tf.keras.datasets.mnist

# Load data (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape training set to (num_samples, height, width, channels)
x_test = x_test.reshape(-1, 28, 28, 1)    # Reshape test set to (num_samples, height, width, channels)
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to be between 0 and 1

def visualize(index):
    plt.imshow(x_train[index])
    plt.show()

model = Sequential([
    Conv2D(32,(3,3),activation = "relu",input_shape = (28,28,1)),
    MaxPooling2D(pool_size = (2,2)),
    Flatten(),
    Dense(300, activation="relu"),
    Dense(len(set(y_train)), activation="softmax")
])

model.compile(optimizer=Adam(),
              loss = "sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.fit(x_train,y_train, epochs = 5)
print(model.evaluate(x_test,y_test))
