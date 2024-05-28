from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy
from tensorflow.keras.preprocessing import image
import keras


imagegen = ImageDataGenerator(rescale=1./255., rotation_range=30, horizontal_flip=True, validation_split=0.1)

# get file path name, create a path, and create variables that connect to train, test, and validation files
path_to_zip = os.getcwd() + '\\shapeDataFull'
PATH = os.path.join(os.path.dirname(path_to_zip), 'shapeDataFull')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

#create gens
train_generator = imagegen.flow_from_directory(train_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(50, 50))
validation_generator = imagegen.flow_from_directory(validation_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(50, 50))
test_generator = imagegen.flow_from_directory(test_dir, class_mode="categorical", shuffle=False, batch_size=128, target_size=(50, 50))

# takes in a path and spits out an image
def returnImageFromPath(path):
    return image.load_img(path)

