# from __future__ import print_function
# import keras
# from keras.datasets import fashion_mnist, cifar10
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# import keras.optimizers as opt
# import keras.regularizers as reg
#
# batch_size = 128
# # batch_size = 256
# # batch_size = 512
# num_classes = 10
# epochs = 20
# # epochs = 10
# # epochs = 100
#
# # the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))
#
# # no dropout
#
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dense(512, activation='relu'))
# # model.add(Dense(10, activation='softmax'))
#
# # l1
#
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dense(512, activation='relu', kernel_regularizer=reg.l1(0.01)))
# # model.add(Dense(10, activation='softmax'))
# #
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dense(512, activation='relu', kernel_regularizer=reg.l1(0.1)))
# # model.add(Dense(10, activation='softmax'))
# #
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dense(512, activation='relu', kernel_regularizer=reg.l1(0.005)))
# # model.add(Dense(10, activation='softmax'))
#
# # l2
#
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dense(512, activation='relu', kernel_regularizer=reg.l2(0.01)))
# # model.add(Dense(10, activation='softmax'))
# #
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dense(512, activation='relu', kernel_regularizer=reg.l2(0.1)))
# # model.add(Dense(10, activation='softmax'))
# #
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dense(512, activation='relu', kernel_regularizer=reg.l2(0.005)))
# # model.add(Dense(10, activation='softmax'))
#
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dropout(0.4))
# # model.add(Dense(512, activation='relu'))
# # model.add(Dropout(0.4))
# # model.add(Dense(10, activation='softmax'))
# #
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# # model.add(Dropout(0.8))
# # model.add(Dense(512, activation='relu'))
# # model.add(Dropout(0.8))
# # model.add(Dense(10, activation='softmax'))
#
# # activation functions
# # model.add(Dropout(0.2))
# # model.add(Dense(10, activation='tanh'))
# #
# # model.add(Dropout(0.2))
# # model.add(Dense(10, activation='sigmoid'))
# #
# # model.add(Dropout(0.2))
# # model.add(Dense(10, activation='gelu'))
#
#
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt.RMSprop(),
#               metrics=['accuracy'])
#
# # model.compile(loss='categorical_crossentropy',
# #               optimizer=opt.SGD(),
# #               metrics=['accuracy'])
#
# # model.compile(loss='categorical_crossentropy',
# #               optimizer=opt.Nadam(),
# #               metrics=['accuracy'])
#
# # model.compile(loss='categorical_crossentropy',
# #               optimizer=opt.Adam(),
# #               metrics=['accuracy'])
#
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size, epochs=epochs,
#                     verbose=1, validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
#
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.optimizers as opt
import keras.regularizers as reg

# Define lists of hyperparameters to iterate over
batch_size = 128
epochs = 20
dropout_values = [0.2, 0.4, 0.8]
regularization_values = [0.01, 0.1, None]
activation_functions = ['relu', 'tanh', 'sigmoid', 'gelu']
optimizers = [opt.SGD, opt.Adam]
num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=opt.RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print("INITIALLY:")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Iterate over batch sizes, epochs, dropout values, regularization methods, and activation functions
for dropout in dropout_values:
    for reg_value in regularization_values:
        for activation_function in activation_functions:
            for optimizer in optimizers:
                print("Training with batch_size = 128, epochs = 20, "
                      f"dropout={dropout}, regularization={reg_value}, "
                      f"activation={activation_function}", f"optimizer = {optimizer}")

                model = Sequential()
                model.add(Dense(512, activation=activation_function, input_shape=(784,),
                                kernel_regularizer=reg.l2(reg_value)))
                model.add(Dropout(dropout))
                model.add(Dense(512, activation=activation_function, kernel_regularizer=reg.l2(reg_value)))
                model.add(Dropout(dropout))
                model.add(Dense(10, activation='softmax'))

                model.summary()

                model.compile(loss='categorical_crossentropy',
                              optimizer=opt.RMSprop(),
                              metrics=['accuracy'])

                history = model.fit(x_train, y_train,
                                    batch_size=batch_size, epochs=epochs,
                                    verbose=1, validation_data=(x_test, y_test))
                score = model.evaluate(x_test, y_test, verbose=0)

                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
                print("")
