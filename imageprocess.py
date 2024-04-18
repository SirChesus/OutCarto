from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


imagegen = ImageDataGenerator(rescale=1./255., rotation_range=30, horizontal_flip=True, validation_split=0.1)

# get file path name, create a path, and create variables that connect to train, test, and validation files
path_to_zip = '/Users/mayankkattela/Downloads/weatherData'
PATH = os.path.join(os.path.dirname(path_to_zip), 'weatherData')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

#create gens
train_generator = imagegen.flow_from_directory(train_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(224, 224))
validation_generator = imagegen.flow_from_directory(validation_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(224, 224))
test_generator = imagegen.flow_from_directory(test_dir, class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

#-----------------------------------------------------------------------

examples_datagen = ImageDataGenerator(rescale=1./255)
examples_generator = examples_datagen.flow_from_directory(PATH, target_size=(224,224), classes=['examples'], shuffle=False)

examples_generator.reset()
pred = model.predict(examples_generator)

predicted_classes = []

for p in pred:
  predicted_class = -1
  max = -1
  for i in range(len(p)):
    if p[i] > max:
      max = p[i]
      predicted_class = i
  predicted_classes.append(predicted_class)