#BASIC CODE, NEEDS WORK
#Nested cross fold validation

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from imageprocess import train_generator, validation_generator
from sklearn.model_selection import KFold
import numpy as np

print(3)


