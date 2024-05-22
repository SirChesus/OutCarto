from cnnMain import model
import imageprocess
#-----------------------------------------------------------------------
#examples_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
#examples_generator = examples_datagen.flow_from_directory(imageprocess.PATH, target_size=(224,224), classes=['examples'], shuffle=False)
example_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
examples_generator = example_datagen.flow_from_directory(imageprocess.test_dir, target_size=(50,50), shuffle=False)


examples_generator.reset()
pred = model.predict(examples_generator)

predicted_classes = []
print('---- ', model.predict_on_batch(examples_generator[0]))

for p in pred:
  predicted_class = -1
  max = -1
  for i in range(len(p)):
    if p[i] > max:
      max = p[i]
      predicted_class = i
  predicted_classes.append(predicted_class)