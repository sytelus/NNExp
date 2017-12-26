from __future__ import print_function
import numpy
numpy.random.seed(42)
#import cntk.device as cntk_device
#cntk_device.try_set_default_device(cntk_device.cpu())
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras import backend as K

batch_size = 10
num_classes = 10
epochs = 500

# the data, shuffled and split between train and test sets
(x_train_all, y_train_all), (x_test_official, y_test_official) = mnist.load_data()
x_train_all = x_train_all.reshape(60000, 784).astype('float32') / 255
x_test_official = x_test_official.reshape(10000, 784).astype('float32') / 255
y_test_official = keras.utils.to_categorical(y_test_official, num_classes).astype('float32')

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.995, stratify=y_train_all)
y_train = keras.utils.to_categorical(y_train, num_classes).astype('float32')
y_test = keras.utils.to_categorical(y_test, num_classes).astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def sum_squared_error(y_true, y_pred):
  return 0.5 * K.sum(K.square(y_pred - y_true), axis=-1)

model = Sequential()
rnd_normal_init = keras.initializers.glorot_normal()
#rnd_normal_init = keras.initializers.he_normal()
#rnd_normal_init = keras.initializers.random_normal(0, 1)
model.add(Dense(300, activation='relu', 
                kernel_initializer=rnd_normal_init, 
                bias_initializer=rnd_normal_init, input_shape=(784,)))
model.add(Dense(100, activation='relu', 
                kernel_initializer=rnd_normal_init, 
                bias_initializer=rnd_normal_init))
model.add(Dense(num_classes, activation='softmax', 
                kernel_initializer=rnd_normal_init, 
                bias_initializer=rnd_normal_init))

model.summary()

model.compile(loss='categorical_crossentropy', #sum_squared_error
              optimizer=keras.optimizers.SGD(lr=0.1),
              metrics=['categorical_accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test_official, y_test_official))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])