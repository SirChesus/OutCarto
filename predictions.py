from cnnMain import model
import imageprocess
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import numpy as np
from matplotlib.widgets import Button



#-----------------------------------------------------------------------
#examples_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
#examples_generator = examples_datagen.flow_from_directory(imageprocess.PATH, target_size=(224,224), classes=['examples'], shuffle=False)
example_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
#examples_generator = example_datagen.flow_from_directory(imageprocess.PATH, target_size=(50,50), shuffle=False)
examples_generator = keras.utils.image_dataset_from_directory(imageprocess.test_dir, image_size=(50,50), shuffle=False)


# Load the image using Matplotlib
#image = mpimg.imread(examples_generator.file_paths[0])
# Display the image
#plt.imshow(image)
#plt.axis('off')  # Turn off axis
#plt.show()

#for y in examples_generator.file_paths:
#  myobj = plt.imshow(mpimg.imread(y))
#  myobj.set_data(y)  

images = []
for i in examples_generator.file_paths:
  images.append(mpimg.imread(i))

# Set up the figure and axis
fig, ax = plt.subplots()
img = ax.imshow(images[0], cmap='gray')  # Initial image
# Initial image index
current_index = 0

# Function to update the displayed image
def update_image(index):
    img.set_data(images[index])
    ax.set_title(f'Image {index + 1}')
    plt.draw()

# Callback function for the button
def next_image(event):
    global current_index
    current_index = (current_index + 1) % len(images)
    update_image(current_index)

# Set up the figure and axis
fig, ax = plt.subplots()
img = ax.imshow(images[current_index], cmap='gray')  # Initial image
ax.set_title(f'Image {current_index + 1}')

# Create the "Next" button
ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
btn_next = Button(ax_button, 'Next')
btn_next.on_clicked(next_image)

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