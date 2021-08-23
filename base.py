import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import matplotlib.pyplot as plt


directory = '/home/matt/Desktop/repos/corgi-detector/images/Images/'
batch_size = 32
epochs = 50

training_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    validation_split=0.2,
    subset="training",
    seed=121,
    batch_size = batch_size,
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    validation_split=0.2,
    subset="validation",
    seed=121,
    batch_size=batch_size,
)

# shapes = training_data.take(1)
# print(shapes(1))

plt.figure(figsize=(10, 10))
for images, labels in training_data.take(1):
    for i in range(30):
        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    training_data, epochs=epochs, validation_data=validation_data,
)

