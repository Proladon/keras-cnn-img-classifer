import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg19
from keras.backend import relu
from keras.losses import SparseCategoricalCrossentropy

classes = ['maid', 'bunny']
imageSize = (500, 500)

train_path = 'C:/Users/Proladon/Desktop/ml/train'
test_path = 'C:/Users/Proladon/Desktop/ml/test'
valid_path = 'C:/Users/Proladon/Desktop/ml/valid'

train_batches = ImageDataGenerator(preprocessing_function = vgg19.preprocess_input)
train_batches = train_batches.flow_from_directory(directory=train_path, 
                                                                          target_size=imageSize, 
                                                                          classes=classes, 
                                                                          batch_size=20)

test_batches = ImageDataGenerator(preprocessing_function = vgg19.preprocess_input)
test_batches = test_batches.flow_from_directory(directory=test_path, 
                                                                          target_size=imageSize, 
                                                                          classes=classes, 
                                                                          batch_size=20)

valid_batches = ImageDataGenerator(preprocessing_function = vgg19.preprocess_input)
valid_batches = valid_batches.flow_from_directory(directory=valid_path, 
                                                                          target_size=imageSize, 
                                                                          classes=classes, 
                                                                          batch_size=20)



model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation=relu, padding='same', input_shape=(500, 500, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=32, kernel_size=(3,3), activation=relu, padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

model.summary()

model.compile(
  optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
)


model.fit(train_batches,
          validation_data=valid_batches,
          epochs=20,
          verbose=2)
model.save_weights('bottleneck_fc_model.h5')