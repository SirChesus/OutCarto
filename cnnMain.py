from keras.layers import Lambda, SeparableConv2D, BatchNormalization, Dropout, MaxPooling2D, Input, Dense, Conv2D, Activation, Flatten 
from keras.models import Sequential
import keras
from imageprocess import train_generator, validation_generator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


""""

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
"""

model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

x = model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(226, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(226, activation='relu')(x)

x = Flatten()(x)

x = Dense(126)(x)
x = Activation('relu')(x)

x = Dense(64)(x)
x = Activation('relu')(x)

"""x = model.output

model.add(GlobalAveragePooling2D())

model.add(Dense(512, Activation = 'relu', Dropout = 0.5))
model.add(GlobalAveragePooling2D())

model.add(Dense(226, Activation = 'relu', Dropout = 0.5))
model.add(Dense(226, Activation = 'relu'))

model.add(Flatten())

model.add(Dense(126))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.CategoricalFocalCrossentropy(), optimizer=keras.optimizers.AdamW(learning_rate=2e-3, weight_decay=.005, use_ema=True), metrics = ['accuracy'])
model.fit(train_generator, epochs = 20, validation_data= validation_generator)"""