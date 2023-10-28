from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.optimizers as opt
import keras.regularizers as reg
from keras.src.layers import Flatten

batch_size = 128
num_classes = 10
epochs = 20
optimizers = [opt.Adam, opt.SGD]

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))  # Flatten the input images
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# no dropout

# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# l1

# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu', kernel_regularizer=reg.l1(0.01)))
# model.add(Dense(10, activation='softmax'))
#
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu', kernel_regularizer=reg.l1(0.1)))
# model.add(Dense(10, activation='softmax'))
#
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu', kernel_regularizer=reg.l1(0.005)))
# model.add(Dense(10, activation='softmax'))

# l2
#
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu', kernel_regularizer=reg.l2(0.01)))
# model.add(Dense(10, activation='softmax'))
#
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu', kernel_regularizer=reg.l2(0.1)))
# model.add(Dense(10, activation='softmax'))
#
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu', kernel_regularizer=reg.l2(0.005)))
# model.add(Dense(10, activation='softmax'))

# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.4))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(10, activation='softmax'))
#
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.8))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.8))
# model.add(Dense(10, activation='softmax'))

# activation functions
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='tanh'))
#
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
#
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='gelu'))


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=opt.RMSprop(),
              metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy',
#               optimizer=opt.SGD(),
#               metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy',
#               optimizer=opt.Nadam(),
#               metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy',
#               optimizer=opt.Adam(),
#               metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
