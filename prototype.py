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


batch_size = 32
img_height = 256
img_width = 256


# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   tf.keras.layers.Conv2D(32, 2, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 2, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 2, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 2, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(2048, activation='relu'),
#   tf.keras.layers.Dense(2048, activation='relu'),
#   tf.keras.layers.Dense(10)
# ])

model= tf.keras.models.load_model("./testingmodel/")

def prepare(frame):
  new_array = cv2.resize(frame,(img_height, img_width))
  return new_array.reshape(-1, img_height, img_width, 3)

def getprediction(prediction):
  key = ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
  maxval = max(prediction[0])
  # print(maxval, prediction)
  index = np.where(prediction[0] == maxval)
  # print(index)
  # print("image value: ", key[int(index[0])])
  return key[int(index[0])]


# frame = cv2.imread("./WIN_20220301_10_15_05_Pro.jpg")
# prediction = model.predict([prepare(frame)])
# print(prediction)
# getprediction(prediction)
while True:
  cap = cv2.VideoCapture("./20220227_091214.mp4")
  fps = 30
  window = [0] * fps
  count = 0
  currough = 0

  while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
      prediction = model.predict([prepare(frame)])
      temp = getprediction(prediction)
      window.pop(0)
      window.append(int(temp))
      if count % (fps/2) == 0:
        print("roughness: ", sum(window)/fps)
      if count >= fps:
        count = 0
      count += 1
      # gray = cv2.cvtColor(prepare(frame),)
      cv2.imshow('frame', cv2.resize(frame,(img_height, img_width)))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      break