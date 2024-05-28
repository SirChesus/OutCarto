from cnnMain import model
import imageprocess
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import numpy as np
from matplotlib.widgets import Button

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

def load_and_preprocess_image(img_path, target_size):
    """
    Load an image from the file path and preprocess it.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


#-----------------------------------------------------------------------
#examples_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
#examples_generator = examples_datagen.flow_from_directory(imageprocess.PATH, target_size=(224,224), classes=['examples'], shuffle=False)
example_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
#examples_generator = example_datagen.flow_from_directory(imageprocess.PATH, target_size=(50,50), shuffle=False)
examples_generator = keras.utils.image_dataset_from_directory(imageprocess.test_dir, image_size=(50,50), shuffle=False)


# creates a list of images
images = []
for i in examples_generator.file_paths:
  # need to convert file path to image and then append it to the image list
  images.append(mpimg.imread(i))

# Initial image index, sent to negative one since on startup calls nextImage
current_index = -1

# creating global text variable so we can just change the data of that variable
text = False

# moves to next shape in dataset
def next_shape(event):
  global current_index
  numberOfImages = len(list(examples_generator.file_paths))
  numberOfClasses = len(list(imageprocess.train_generator.class_indices.items()))
  # only works if each train thing is the same
  imagesPerClass = numberOfImages / numberOfClasses
  # multiplying currentIndex * # of different classes / total length to then add 1 and multiply by reciporacol
  #print(current_index)
  #print(len(list(imageprocess.train_generator.class_indices.items())))
  #print(len(list(examples_generator.file_paths)))
  #print(current_index * len(list(imageprocess.train_generator.class_indices.items())) / len(list(examples_generator.file_paths)) + 1)
  #print(" * ", (len(list(examples_generator.file_paths)) / len(list(imageprocess.train_generator.class_indices.items()))))
  #current_index = (int)(current_index * len(list(imageprocess.train_generator.class_indices.items())) * (len(list(examples_generator.file_paths)) + 1) * (int)(len(list(examples_generator.file_paths)) / len(list(imageprocess.train_generator.class_indices.items()))))
  #print(current_index)

  #current_index = int(current_index - current_index % imagesPerClass + imagesPerClass)
  
  fileShape = examples_generator.file_paths[current_index].split("/")
  fileShape = fileShape[len(fileShape)-1].split("_")[0]
  print(fileShape)

  for i in range(current_index, len(examples_generator.file_paths)):
    #print(examples_generator.file_paths[i])
    if (fileShape in examples_generator.file_paths[i]):
      current_index += 1
  current_index = current_index % numberOfImages
  
  print(current_index)
  update_image(current_index, predictOnModel())


# Function to update the displayed image
def update_image(index, prediction):
  global text
  img.set_data(images[index])
  ax.set_title(f'Image {index + 1}')
  text.set_text(prediction)
  plt.draw()

def predictOnModel():
  # getting image based on path
  img_path = examples_generator.file_paths[current_index]

  # Load and preprocess the image
  img_array = load_and_preprocess_image(img_path, (50, 50))
  prediction = model.predict(img_array)
  # converting the prediction to the highest guess
  return list(imageprocess.train_generator.class_indices.items())[np.argmax(prediction)][0]


# Callback function for the button
def next_image(event):
  global current_index
  current_index = (current_index + 1) % len(images)
  
  update_image(current_index, predictOnModel())
    

# Set up the figure and axis
fig, ax = plt.subplots()
img = ax.imshow(images[current_index+1], cmap='gray')  # Initial image
text = ax.text(0.2,10, "empty", fontsize = 10, color = 'black')
next_image(None)


ax.set_xticks([])
ax.set_yticks([])

# Create the "Next" button
ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])

btn_next = Button(ax_button, 'Next')
btn_next.on_clicked(next_image)

btn_next_shape = Button(plt.axes([.81,.15,.25,.07]), 'Next Shape')
btn_next_shape.on_clicked(next_shape)


# Display the plot
plt.show()


#print(len(examples_generator))
#for i in range(len(examples_generator.file_paths)):
  #print("file name is " + examples_generator.file_paths[i])

#examples_generator.reset()
#pred = model.predict(examples_generator)

#predicted_classes = []

#for p in pred:
  #predicted_class = -1
  #max = -1
  #for i in range(len(p)):
    #if p[i] > max:
      #max = p[i]
      #predicted_class = i
  #predicted_classes.append(predicted_class)