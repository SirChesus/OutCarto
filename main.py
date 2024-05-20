from imageprocess import *
from cnnMain import *

print(' ')
# printing predictions 
for i in range(len(examples_generator.filenames)):
  print(' ')
  print("File Name: ",str(examples_generator.filenames[i]))
  print("Label: ", list(train_generator.class_indices.items())[i][0])