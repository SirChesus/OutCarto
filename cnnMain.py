from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.models import Sequential
import keras
from imageprocess import train_generator, validation_generator

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(50,50, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


#drops values during training at rate .25 ---> prevent overfitting
model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.CategoricalFocalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate= 1e-2, weight_decay=0.002), metrics = ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)])
model.fit(train_generator, epochs = 3, validation_data= validation_generator)