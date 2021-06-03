import tensorflow as tf
import os
import random
import glob
import shutil
import matplotlib.pyplot as plt


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('GPU:', len(physical_devices))


# os.chdir('C:/Users/Proladon/Desktop/ml')

# for _ in random.sample(glob.glob('maid*'), 80):
#   shutil.move(_, 'train/maid')
  
# for _ in random.sample(glob.glob('bunny*'), 80):
#   shutil.move(_, 'train/bunny')
  
  
# for _ in random.sample(glob.glob('maid*'), 40):
#   shutil.move(_, 'test/maid')
  
# for _ in random.sample(glob.glob('bunny*'), 40):
#   shutil.move(_, 'test/bunny')
  

# for _ in random.sample(glob.glob('maid*'), 20):
#   shutil.move(_, 'valid/maid')
  
# for _ in random.sample(glob.glob('bunny*'), 20):
#   shutil.move(_, 'valid/bunny')
  
train_path = 'C:/Users/Proladon/Desktop/ml/train'
test_path = 'C:/Users/Proladon/Desktop/ml/test'
valid_path = 'C:/Users/Proladon/Desktop/ml/valid'

train_batches = tf.keras.preprocessing.image.ImageDataGenerator(
  preprocessing_function = tf.keras.applications.vgg19.preprocess_input).flow_from_directory(directory=train_path, target_size=(500, 500), classes=['maid', 'bunny'], batch_size=20)

test_batches = tf.keras.preprocessing.image.ImageDataGenerator(
  preprocessing_function = tf.keras.applications.vgg19.preprocess_input).flow_from_directory(directory=test_path, target_size=(500, 500), classes=['maid', 'bunny'], batch_size=20)

valid_batches = tf.keras.preprocessing.image.ImageDataGenerator(
  preprocessing_function = tf.keras.applications.vgg19.preprocess_input).flow_from_directory(directory=valid_path, target_size=(500, 500), classes=['maid', 'bunny'], batch_size=20)


assert train_batches.n == 160
assert test_batches.n == 80
assert valid_batches.n == 40

imgs, labels = next(train_batches)

def plotImages(images_arr):
  fig, axes = plt.subplots(1, 10, figsize=(20,20))
  axes = axes.flatten()
  for img, ax in zip( images_arr, axes ):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()
  
plotImages(imgs)
print(labels)