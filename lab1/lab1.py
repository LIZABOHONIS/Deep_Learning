import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Початкові значення пікселів 0 - 255, нормалізуємо їх в діапазон 0 - 1
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochsCount = 50
history = model.fit(train_images, train_labels, epochs=epochsCount)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('\nТочність на тестових даних:', test_acc)

#Функція витрат
plt.plot(np.arange(0, epochsCount), history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

#Функція для побудови графіку результату передбачення
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
  plt.show()

def plot_image(predictions_array, true_label, img):
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
  plt.show()


def test():
  results = {'T-shirt/top': [0,0], 'Trouser':[0,0], 'Pullover':[0,0], 'Dress':[0,0], 'Coat':[0,0],
               'Sandal':[0,0], 'Shirt':[0,0], 'Sneaker':[0,0], 'Bag':[0,0], 'Ankle boot':[0,0]}
  for i in range(0,900,450):
    img = test_images[i]
    lable = test_labels[i]
    img = (np.expand_dims(img,0))
    predictions_single = model.predict(img)
    predictedLabel = np.argmax(predictions_single[0])

    data = results[class_names[lable]]
    if predictedLabel == lable:
      data[1] += 1
    else:
      data[0] += 1
    results[lable] = data
    print(i)
  for key in results:
    data = results[key]
    persents = (data[1] * 100)/(data[0] + data[1])
    print(key, ": ", persents, "% success")


test()

# Берем одну картинку з тестового сету.
imageIndex =450;
img = test_images[imageIndex]
lable = test_labels[imageIndex]
# Додаємо зображення в пакет даних, що складається тільки з одного елемента.
img = (np.expand_dims(img,0))
predictions_single = model.predict(img)

plot_image(predictions_single, lable, img)
plot_value_array(predictions_single,lable)
