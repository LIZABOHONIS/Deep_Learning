import tensorflow as tf
from tensorflow import keras
import streamlit as st
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Початкові значення пікселів 0 - 255, нормалізуємо їх в діапазон 0 - 1
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

def create_model():
  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  return model
def load_model():
  model = create_model()
  try:
      model.load_weights(checkpoint_path)
  except:
    pass
  return model

def Train(epochsCount):
    history = model.fit(train_images, train_labels, epochs=epochsCount, callbacks=[cp_callback])
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
    plt.plot(np.arange(0, epochsCount), history.history["loss"])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    return test_acc

# Функція для побудови графіку результату передбачення
def plot_value_array(predictions_array, true_label):
  predictions_array = predictions_array[0]
  plt.grid(False)
  plt.xticks(range(10), class_names, rotation=45)
  plt.yticks(predictions_array)
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  st.pyplot()

def plot_image(i, predictions_array, true_label, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[0], cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("Predicted:{} {:2.0f}% (Real:{})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
  st.pyplot()


model = load_model()
st.text('Dataset Fashion MNIST')
epochs = st.slider('Choose epochs', 5, 50)
if st.button('Train network'):
    st.write('\nAccuracy on test data:', 100*Train(epochs),'%')
    st.write('Cost function')
    st.pyplot()

# Берем одну картинку з тестового сету.
imageIndex = st.slider('Select image from the test set', 0, 9999)
if st.button('Make prediction'):
  img = test_images[imageIndex]
  lable = test_labels[imageIndex]
  # Додаємо зображення в пакет даних, що складається тільки з одного елемента.
  img = (np.expand_dims(img,0))
  predictions_single = model.predict(img)
  plot_image(imageIndex, predictions_single, lable, img)
  plot_value_array(predictions_single,lable)
