# # Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Convolution2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense, Dropout
# import os
# # import scipy
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sz = 128
# # Step 1 - Building the CNN
#
# # Initializing the CNN
# classifier = Sequential()
#
# # First convolution layer and pooling
# classifier.add(Convolution2D(28, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Second convolution layer and pooling
# classifier.add(Convolution2D(28, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Convolution2D(28, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
# # Flattening the layers
# classifier.add(Flatten())
#
# # Adding a fully connected layer
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dropout(0.40))
# classifier.add(Dense(units=96, activation='relu'))
# classifier.add(Dropout(0.40))
# classifier.add(Dense(units=64, activation='relu'))
# classifier.add(Dense(units=24, activation='softmax')) # softmax for more than 2
#
# # Compiling the CNN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2
#
#
# # Step 2 - Preparing the train/test data and training the model
# classifier.summary()
# # Code copied from - https://keras.io/preprocessing/image/
# from keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
# batch_size = 5
# training_set = train_datagen.flow_from_directory('data2/train',
#                                                  target_size=(sz, sz),
#                                                  batch_size=batch_size,
#                                                  color_mode='grayscale',
#                                                  class_mode='categorical')
#
# test_set = test_datagen.flow_from_directory('data2/test',
#                                             target_size=(sz, sz),
#                                             batch_size=batch_size,
#                                             color_mode='grayscale',
#                                             class_mode='categorical')
#
# steps_per_epoch_train = len(training_set) // batch_size
# steps_per_epoch_test = len(test_set) // batch_size
#
# classifier.fit(
#         training_set,
#         steps_per_epoch=steps_per_epoch_train, # No of images in training set
#         epochs=5,
#         validation_data=test_set,
#         validation_steps=steps_per_epoch_test)# No of images in test set
#
#
# # Saving the model
# model_json = classifier.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# print('Model Saved')
# classifier.save_weights('model-bw.h5')
# print('Weights saved')














import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define your data directory path
data_dir = 'data2/train'

# Define data augmentation parameters (optional but can improve model generalization)
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into train and validation sets
)

# Create separate generators for training and validation data

train_generator = datagen.flow_from_directory(
                                                 'data2/train',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 color_mode='grayscale',
                                                 class_mode='categorical'
)
validation_generator = datagen.flow_from_directory(
                                                 'data2/test',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 color_mode='grayscale',
                                                 class_mode='categorical'
)

# Create a CNN model (you can modify this architecture)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(24, activation='softmax')  # Adjust the output units based on your problem
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,  # Adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Evaluate the model
accuracy = model.evaluate(validation_generator)[1]
print(f"Validation accuracy: {accuracy * 100:.2f}%")

# Save the model in h5 and JSON formats
model.save('model-bw.h5')
model_json = model.to_json()
with open('model-bw.json', 'w') as json_file:
    json_file.write(model_json)
