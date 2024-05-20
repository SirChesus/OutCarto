#BASIC CODE, NEEDS WORK
#Nested cross fold validation

from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from imageprocess import train_generator, validation_generator
from sklearn.model_selection import KFold
import numpy as np

# Merge inputs and targets
inputs = np.concatenate(train_generator, axis=0)
targets = np.concatenate(validation_generator, axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=4, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
  # Define the model architecture
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # drops values during training at rate .25 ---> prevent overfitting
  model.add(Dropout(.25))
  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(4))
  model.add(Activation('softmax'))

  # Compile the model
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1


