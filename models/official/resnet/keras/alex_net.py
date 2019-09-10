from tensorflow.python.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda

def alex_net(num_classes, dtype='float32', batch_size=None):

    bn_axis = 3
    data_format ='channels_last'
    input_shape = (224, 224, 3)


    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=4,
                         padding='same', input_shape=input_shape,
                         data_format=data_format))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same',
                         data_format=data_format))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, data_format=data_format))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same',
                         data_format=data_format))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, data_format=data_format))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same', data_format=data_format))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, data_format=data_format))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

