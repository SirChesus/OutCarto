from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


imagegen = ImageDataGenerator(rescale=1./255., rotation_range=30, horizontal_flip=True, validation_split=0.1)

# get file path name, create a path, and create variables that connect to train, test, and validation files
path_to_zip = 'C:\\Users\\Test0\\Downloads\\weatherData-20240507T192738Z-001\\weatherData'
PATH = os.path.join(os.path.dirname(path_to_zip), 'weatherData')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

#create gens
train_generator = imagegen.flow_from_directory(train_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(224, 224))
validation_generator = imagegen.flow_from_directory(validation_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(224, 224))
test_generator = imagegen.flow_from_directory(test_dir, class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))