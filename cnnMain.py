from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.models import Sequential
import keras
from imageprocess import train_generator, validation_generator

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(50,50, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(124, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

<<<<<<< Updated upstream
model.add(Conv2D(64,(3,3)))
=======
model.add(Conv2D(124, (3,3)))
>>>>>>> Stashed changes
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


#drops values during training at rate .25 ---> prevent overfitting
model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(122))
model.add(Activation('relu'))

model.add(Dense(122))
model.add(Activation('relu'))

<<<<<<< Updated upstream
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
=======
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.CategoricalFocalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate= 1e-4, weight_decay=0.002), metrics = ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)])
>>>>>>> Stashed changes
model.fit(train_generator, epochs = 10, validation_data= validation_generator)