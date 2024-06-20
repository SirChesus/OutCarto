from cnnMain import model
import imageprocess
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import numpy as np
from matplotlib.widgets import Button
from PIL import Image


import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

def load_and_preprocess_image(img_path, target_size=(100, 100, 3)):
    # Load the image with PIL
    img = Image.open(img_path)
    
    # Convert the image to grayscale
    #img = img.convert('L')
    
    # Resize the image to the target size
    img = img.resize(target_size)
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Add an extra dimension to match the expected input shape (50, 50, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    
    # Expand dimensions to match the batch size
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to the range [0, 1]
    img_array = img_array / 255.0
    
    return img_array


#-----------------------------------------------------------------------
#examples_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
#examples_generator = examples_datagen.flow_from_directory(imageprocess.PATH, target_size=(224,224), classes=['examples'], shuffle=False)
example_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
#examples_generator = example_datagen.flow_from_directory(imageprocess.PATH, target_size=(50,50), shuffle=False)
examples_generator = keras.utils.image_dataset_from_directory(imageprocess.test_dir, image_size=(100,100), shuffle=False)


# creates a list of images
images = []
for i in examples_generator.file_paths:
  # need to convert file path to image and then append it to the image list
  images.append(mpimg.imread(i))
  # could be much more efficient
  #images.reverse()


# Initial image index, sent to negative one since on startup calls nextImage
current_index = -1

# creating global text variable so we can just change the data of that variable
text = False

# moves to next shape in dataset
def next_shape(event):
  global current_index
  # gets all the images, not sure if file_paths is needed if there is an easier way to convert to an array
  numberOfImages = len(list(examples_generator.file_paths))

  # gets the last folder that the file path is in, should be the class **TODO** fix comment
  fileShape = examples_generator.file_paths[current_index].split("\\")
  fileShape = fileShape[len(fileShape)-2]

  # loops through all the images from current index adding +1 to current index for each file that's name contains the shape
  for i in range(current_index, len(examples_generator.file_paths)):
    #print(examples_generator.file_paths[i])
    if (fileShape in examples_generator.file_paths[i]):
      current_index += 1
  
  # preventing overflow errors, will just loop around
  current_index = current_index % numberOfImages
  
  # updates the display with a new prediction and the current index
  update_image(current_index, predictOnModel())


# Function to update the displayed image
def update_image(index, prediction):
  global text

  image = mpimg.imread(examples_generator.file_paths[current_index])
  image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

  img.set_data(image)

  #img.set_data(Image.open(examples_generator.file_paths[index]).convert("L"))

  # setting the title to the file name
  fileShape = examples_generator.file_paths[current_index].split("\\")
  fileShape = fileShape[len(fileShape)-1]
  ax.set_title(fileShape, fontsize = 7)

  # setting the text so no new object is created
  text.set_text(prediction)
  plt.draw()

def predictOnModel():
  # getting image based on path
  img_path = examples_generator.file_paths[current_index]
  print("path it is predicting on", img_path)

  # Load and preprocess the image
  img_array = load_and_preprocess_image(img_path, target_size=(200,200))
  prediction = model.predict(img_array)
  # converting the prediction to the highest guess
  return list(imageprocess.train_generator.class_indices.items())[np.argmax(prediction)][0]

# trains the ai for one more epoch



# Call back next button
def next_image(event):
  global current_index
  current_index = (current_index + 1) % len(images)
  
  update_image(current_index, predictOnModel())
    
# Call back previous button
def previous_image(event):
  global current_index
  current_index = (current_index - 1) % len(images)
  update_image(current_index, predictOnModel())

# Set up the figure and axis
fig, ax = plt.subplots()
img = ax.imshow(images[current_index+1])  # Initial image
next_image(None)

text = plt.axes([.1,.8,.2,.2], "empty", fontsize = 6, color = 'black')



ax.set_xticks([])
ax.set_yticks([])

# Create the "Next" button
ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])

btn_next = Button(ax_button, 'Next')
btn_next.on_clicked(next_image)

btn_next_shape = Button(plt.axes([.81,.15,.25,.07]), 'Next Shape')
btn_next_shape.on_clicked(next_shape)

btn_previous = Button(plt.axes([.81,.25,.1,.20]), 'Previous')
btn_previous.on_clicked(previous_image)

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