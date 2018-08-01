from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras.utils import to_categorical

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocess data
if K.image_data_format() == 'channels_last':
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    input_shape=(784, )
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


print(x_train.shape)
print(x_test.shape)
print(K.image_data_format())
print(y_train[0])

model = Sequential()
model.add(Dense(784, activation='relu', input_shape=input_shape))
model.add(Dropout(.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()


model.fit(x_train, y_train, epochs=12, batch_size=32, validation_data=(x_test, y_test))


