from imageprocess import *
from cnnMain import *
from predictions import examples_generator
from predictions import predicted_classes

print(' ')
# printing predictions 

for i in range(len(examples_generator.filenames)):
  print(' ')
  print("File Name: ",str(examples_generator.filenames[i]))
  print("Label: ", list(train_generator.class_indices.items())[i][0])