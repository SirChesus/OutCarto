from cnnMain import model
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