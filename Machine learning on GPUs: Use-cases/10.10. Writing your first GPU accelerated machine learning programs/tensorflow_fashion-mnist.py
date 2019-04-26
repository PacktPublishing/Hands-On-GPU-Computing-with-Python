from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plot

from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plot.figure()
plot.imshow(train_images[0])
plot.colorbar()
plot.grid(False)
plot.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plot.figure(figsize=(10,10))
for i in range(25):
    plot.subplot(5,5,i+1)
    plot.xticks([])
    plot.yticks([])
    plot.grid(False)
    plot.imshow(train_images[i], cmap=plot.cm.binary)
    plot.xlabel(class_names[train_labels[i]])
plot.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=100)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

print(predictions[0])

print(np.argmax(predictions[0]))

print(test_labels[0])




def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plot.grid(False)
  plot.xticks([])
  plot.yticks([])

  plot.imshow(img, cmap=plot.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plot.grid(False)
  plot.xticks([])
  plot.yticks([])
  thisplot = plot.bar(range(10), predictions_array, color="#777777")
  plot.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plot.figure(figsize=(6,3))
plot.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plot.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plot.show()



i = 12
plot.figure(figsize=(6,3))
plot.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plot.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plot.show()


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plot.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plot.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plot.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plot.show()



# Grab an image from the test dataset
img = test_images[0]

print(img.shape)


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)



plot_value_array(0, predictions_single, test_labels)
_ = plot.xticks(range(10), class_names, rotation=45)
plot.show()

print(np.argmax(predictions_single[0]))


