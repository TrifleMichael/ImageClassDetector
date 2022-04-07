import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# ------------ settings ---------------

EPOCHS = 12
DATA_NAME = "newData"

# ------------ datasets ------------

batch_size = 32
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_NAME,
  validation_split=0.2,
  subset="training",
  seed=123,
  batch_size=batch_size)

class_names = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_NAME,
  validation_split=0.2,
  subset="validation",
  seed=123,
  batch_size=batch_size)

# ------------ model preparation  ------------

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 6  # 26


data_augmentation = tf.keras.Sequential(
  [
    layers.RandomRotation(0.2)
  ]
)


model = tf.keras.Sequential([
  data_augmentation,
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  layers.Dropout(0.1),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# ------------ compiling model ------------

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

model.save("./MODEL")

# ------------ show results ----------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])
#
# predictions = probability_model.predict(val_ds)
# for images, labels in val_ds:
#   for k in range(10):
#     for i in range(9*k,9*k+9):
#       ax = plt.subplot(3, 3, i - 9*k + 1)
#       plt.imshow(images[i].numpy().astype("uint8"))
#       plt.title(class_names[np.argmax(predictions[i])]+str(round(max(predictions[i])*100,2))+"%")
#       plt.axis("off")
#     plt.show()