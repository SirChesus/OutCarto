from keras.layers import Lambda, SeparableConv2D, BatchNormalization, Dropout, MaxPooling2D, Input, Dense, Conv2D, Activation, Flatten 
from keras.models import Sequential
import keras
from imageprocess import train_generator, validation_generator

model = Sequential()

model.add(Conv2D(128,(3,3), input_shape=(100,100,3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
# removed pooling layer, testing
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))

#drops values during training at rate .25 ---> prevent overfitting
model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(126))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('softmax'))


model.compile(loss=keras.losses.CategoricalFocalCrossentropy(), optimizer=keras.optimizers.AdamW(learning_rate=2e-3, weight_decay=.005, use_ema=True), metrics = ['accuracy'])
model.fit(train_generator, epochs = 20, validation_data= validation_generator)