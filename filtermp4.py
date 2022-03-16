# TensorFlow and tf.keras
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/tools/cuda/bin")

import tensorflow as tf

# access pics
import pathlib
import cv2

#file manipulation

import PIL
import PIL.Image

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

"""# Pre-Process Data

Create a shortcut of this Google Drive folder in your main Google drive directory (a.k.a "My Drive"). The resulting path should be `My Drive/Pictures`. If the directory is placed anywhere else you will need to adjust the `data_dir` string to point to the correct location.

This directory contains the pre-sorted training images of the asphalt at different roughness levels that will be fed into the neural network to generate the model.

https://drive.google.com/drive/folders/1sgRUGI_mXmF1Ue8zNnxHRU0tR5-QgBfU?usp=sharing
"""

data_dir = "./Pictures"
data_dir = pathlib.Path(data_dir)

"""In this step we simply get the number of image files recognized within Colab. The target number, assuming that the above step was done correctly, should be 99."""

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

"""We get the list of all .jpg files in the mounted directory, then render the first in the list to verify that it is not corrupted. 

The rendering code has been commented out as it is quite slow to run. Uncomment it out if you want to try it out!
"""

road = list(data_dir.glob('*/*.jpg'))
# PIL.Image.open(str(road[0]))

"""# Initialize the Neural Network

Here we set up the information for the initial data set. You may notice that the `img_height` and `img_width` variables do not match with the resolution of the input images you will find in the provided `/Pictures/` directory. This is due to both limits on free Google GPU capacity, as well as observed improvements to accuracy when images were compressed to this size.

We initialize the training and validation sets of images (train_ds and val_ds, respectively) by using a 80/20 train to test split. The class names are derived by the names of the folders the images exist in. 

The total number of training pictures divided by the batch size defined how many training runs the model will go through per epoch. Each run takes a batch size number of pictures to train on per run within the epoch. We used 32 for our training which would result in 3 runs of training per epoch.

The autotune variable ensures that the images are cached, speeding up the training. 
"""

batch_size = 32
img_height = 256
img_width = 256


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split= 0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 2, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 2, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 2, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 2, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(2048, activation='relu'),
  tf.keras.layers.Dense(2048, activation='relu'),
  tf.keras.layers.Dense(10)
])

model= tf.keras.models.load_model("./testingmodel/")

def prepare(frame):
  new_array = cv2.resize(frame,(img_height, img_width))
  return new_array.reshape(-1, img_height, img_width, 3)

def getprediction(prediction):
  key = ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
  maxval = max(prediction[0])
  if(maxval < 0):
    print("no road found")
    return
  # print(maxval, prediction)
  index = np.where(prediction[0] == maxval)
  # print(index)
  print("image value: ", key[int(index[0])])


# frame = cv2.imread("./WIN_20220301_10_15_05_Pro.jpg")
# prediction = model.predict([prepare(frame)])
# print(prediction)
# getprediction(prediction)


#skip past count 9769 frames
cap = cv2.VideoCapture("./20220302_133133.mp4")
count = 0
skipcount = 9769
while(cap.isOpened()):
  ret, frame = cap.read()

  if ret == True:

    # gray = cv2.cvtColor(prepare(frame),)
    count += 1
    print(count - 1)
    if count <= skipcount:
      continue
    prediction = model.predict([prepare(frame)])
    getprediction(prediction)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    x = input("value: ")
    cv2.imwrite("./Pictures/" + x + "/frame%d.jpg" % (count - 1), frame)
    

  else:
    break