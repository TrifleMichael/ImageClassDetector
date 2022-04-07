import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# ---------- loading model ---------------------
from Settings import MODEL_PATH, TEST_PATH

img_height = 256
img_width = 256
model = tf.keras.models.load_model(MODEL_PATH)

# ----------- import class names -----------------

f = open("class_names.txt", "r")
class_names_string = f.read()
class_names_string = class_names_string.split("\'")
class_names = []
for i in range(1, len(class_names_string)-1, 2):
    class_names.append(class_names_string[i])

# ----------- model summary ------------------

#model.summary()

# ----------------- predict class of new data ----------------


test_path = TEST_PATH
test_images = [test_path + "/" + name for name in os.listdir(test_path)]
images = []
predictions = []

for image_name in test_images:
    img = tf.keras.utils.load_img(
        image_name, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    images.append(img)

    predictions.append(model.predict(img_array)[0])

scores = []
for prediction in predictions:
    scores.append(tf.nn.softmax(prediction))

for i in range(0, len(images), 9):
    for j in range(i, min(i+9, len(images))):
        plt.subplot(3, 3, j - i + 1)
        plt.imshow(images[j])
        plt.title(class_names[np.argmax(scores[j])] + " " + str(round(max(100 * np.max(scores[j]), 2))) + "%")
        plt.axis("off")
    plt.show()

