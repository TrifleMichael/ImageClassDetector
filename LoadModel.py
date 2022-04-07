import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# ---------- loading model ---------------------

class_names = ['BMW', 'CocaCola', 'Electrolux']
batch_size = 32
img_height = 256
img_width = 256
model = tf.keras.models.load_model('./MODEL')

# ----------- model summary ------------------

#model.summary()


# ----------------- predict class of new data ----------------


test_path = "./testIMG.jpg"

img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

