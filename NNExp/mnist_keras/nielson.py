from __future__ import print_function
import numpy
numpy.random.seed(42)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

batch_size = 10
num_classes = 10
epochs = 30

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def sum_squared_error(y_true, y_pred):
  return 0.5 * K.sum(K.square(y_pred - y_true), axis=-1)

model = Sequential()
rnd_normal_init = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
model.add(Dense(30, activation='sigmoid', 
                kernel_initializer=rnd_normal_init, 
                bias_initializer=rnd_normal_init, input_shape=(784,)))
model.add(Dense(num_classes, activation='sigmoid', 
                kernel_initializer=rnd_normal_init, 
                bias_initializer=rnd_normal_init))

model.summary()

model.compile(loss=sum_squared_error, #'mean_squared_error',
              optimizer=keras.optimizers.SGD(lr=3),
              metrics=['categorical_accuracy', 'accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])