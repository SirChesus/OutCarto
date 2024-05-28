from keras.layers import Lambda, SeparableConv2D, BatchNormalization, Dropout, MaxPooling2D, Input, Dense, Conv2D, Activation, Flatten 
from keras.models import Sequential
import keras
from imageprocess import train_generator, validation_generator

model = Sequential()

#model.add(Conv2D(128,(3,3), input_shape=(50,50, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(64, (3,3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))

#drops values during training at rate .25 ---> prevent overfitting
#model.add(Dropout(.25))

#model.add(Flatten())

#model.add(Dense(64))
#model.add(Activation('relu'))

#model.add(Dense(8))
#model.add(Activation('softmax'))


# stole this model from the link you wanted to see how it was did
# First Conv block
model.add(Conv2D(16 , (3,3) , padding = 'same' , activation = 'relu' , input_shape = (50,50,3)))
model.add(Conv2D(16 , (3,3), padding = 'same' , activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# Second Conv block
model.add(SeparableConv2D(32, (3,3), activation = 'relu', padding = 'same'))
model.add(SeparableConv2D(32, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

# Third Conv block
model.add(SeparableConv2D(64, (3,3), activation = 'relu', padding = 'same'))
model.add(SeparableConv2D(64, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

# Forth Conv block
model.add(SeparableConv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(SeparableConv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

# FC layer 
model.add(Flatten())
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 64 , activation = 'relu'))
model.add(Dropout(0.3))


# Output layer
model.add(Dense(units = 8, activation = 'softmax'))




model.compile(loss=keras.losses.CategoricalFocalCrossentropy(), optimizer=keras.optimizers.Adam(), metrics = ['accuracy'])
model.fit(train_generator, epochs = 50, validation_data= validation_generator)