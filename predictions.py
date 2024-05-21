from cnnMain import model
import imageprocess
#-----------------------------------------------------------------------
test_generator = imageprocess.imagegen.flow_from_directory(imageprocess.test_dir, class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
examples_datagen = imageprocess.ImageDataGenerator(rescale=1./255)
examples_generator = examples_datagen.flow_from_directory(imageprocess.PATH, target_size=(224,224), classes=['examples'], shuffle=False)

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