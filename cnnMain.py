from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from imageprocess import train_generator, validation_generator

# good info https://keras.io/guides/sequential_model/
# https://www.datacamp.com/tutorial/tutorial-gradient-descent 

#sequential has 3 variables when initilizing, layers, trainable, name
# layers is self explanatory, number of layers for the model, trainable is it trainable, and name is just name

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(224,224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#drops values during training at rate .25 ---> prevent overfitting
model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(train_generator, epochs = 10, validation_data= validation_generator)