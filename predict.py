from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Activation, Flatten
from keras.backend import relu
import os

from keras_preprocessing import image
import numpy as np



test_path = 'C:/Users/Proladon/Desktop/ml/predict'

model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation=relu, padding='same', input_shape=(500, 500, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=32, kernel_size=(3,3), activation=relu, padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])
model.load_weights('maid_v_bunny_model.h5')

model.summary()

model.compile(
  optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
)


for i in os.listdir(test_path):
  img = image.load_img(test_path + '//' + i, target_size=(500, 500))
  # imshow(img)
  # show()
  
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  val = model.predict(images)
  print(f"{i}\nmaid: {val[0][0]}\nbunny: {val[0][1]}\n")