from keras.layers import Lambda, SeparableConv2D, BatchNormalization, Dropout, MaxPooling2D, Input, Dense, Conv2D, Activation, Flatten 
from keras.models import Sequential
import keras
from imageprocess import train_generator, validation_generator

model = Sequential()

model.add(Conv2D(128,(3,3), input_shape=(50,50, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#drops values during training at rate .25 ---> prevent overfitting
model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('softmax'))


model.compile(loss=keras.losses.CategoricalFocalCrossentropy(), optimizer=keras.optimizers.Adam(), metrics = ['accuracy'])
model.fit(train_generator, epochs = 20, validation_data= validation_generator)