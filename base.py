import os

from tensorflow.python.ops.gen_array_ops import size 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import matplotlib.pyplot as plt


image_size = (128, 128)

directory = "images\Images"
batch_size = 32

training_data = tf.keras.preprocessing.image_dataset_from_directory(
    "images\Images",
    validation_split=0.2,
    subset="training",
    seed=121,
    image_size=image_size,
    batch_size = batch_size,
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    "images\Images",
    validation_split=0.2,
    subset="validation",
    seed=121,
    image_size=image_size,
    batch_size=batch_size,
)

# shapes = training_data.take(1)
# print(shapes(1))

plt.figure(figsize=(10, 10))
for images, labels in training_data.take(1):
    for i in range(20):
        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")




model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

epochs = 3

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    training_data, epochs=epochs, callbacks=callbacks, validation_data=validation_data,
)


img = keras.preprocessing.image.load_img(
    "infer_corgi.jpg", target_size = image_size
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]

model.summary()
print("--------")
print(
    predictions
)