#!/usr/bin/env python
# coding: utf-8

# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to your training and test data
train_data_dir = "D:/Lecture/ML/CIA 2/All/dogs-vs-cats/train"
test_data_dir = "D:/Lecture/ML/CIA 2/All/dogs-vs-cats/test1"

# Image dimensions and batch size
img_width, img_height = 64, 64
batch_size = 32

# Data augmentation for the training set
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Data augmentation for the test set (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate the training set
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Generate the test set
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)


# In[14]:


from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the user-uploaded image
img_path ="D:/Lecture/ML/CIA 2/testing1.jpg"
img = image.load_img(img_path, target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make a prediction
prediction = model.predict(img_array)
class_label = "Cat" if prediction[0][0] < 0.5 else "Dog"

print(f"The uploaded image is a {class_label}.")


# In[ ]:




