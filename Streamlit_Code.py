#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil

# Title of the Streamlit app
st.title("Cat and Dog Image Classification")

# Create a folder to store user-uploaded files
user_folder = "user_files"
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

# Upload training data folder
train_data_uploaded = st.file_uploader("Upload Training Data Folder (Cat and Dog images)", type="zip")
if train_data_uploaded:
    # Save the uploaded file to the user folder
    train_data_path = os.path.join(user_folder, "train_data.zip")
    with open(train_data_path, "wb") as f:
        f.write(train_data_uploaded.read())
    
    # Extract the contents of the zip file
    with st.spinner("Extracting training data..."):
        shutil.unpack_archive(train_data_path, os.path.join(user_folder, "train_data"))
        st.success("Training data uploaded and extracted successfully!")

# Upload testing data folder
test_data_uploaded = st.file_uploader("Upload Testing Data Folder (Cat and Dog images)", type="zip")
if test_data_uploaded:
    # Save the uploaded file to the user folder
    test_data_path = os.path.join(user_folder, "test_data.zip")
    with open(test_data_path, "wb") as f:
        f.write(test_data_uploaded.read())
    
    # Extract the contents of the zip file
    with st.spinner("Extracting testing data..."):
        shutil.unpack_archive(test_data_path, os.path.join(user_folder, "test_data"))
        st.success("Testing data uploaded and extracted successfully!")

# Upload an image for prediction
user_image = st.file_uploader("Upload an Image for Prediction", type=["jpg", "jpeg", "png"])
if user_image:
    # Save the uploaded image to the user folder
    img_path = os.path.join(user_folder, "user_image.jpg")
    with open(img_path, "wb") as f:
        f.write(user_image.read())

    # Display the uploaded image
    st.image(user_image, caption="Uploaded Image", use_column_width=True)

    # Load the pre-trained model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_generator, epochs=10)  # You may need to define train_generator
    
    # Make a prediction
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    class_label = "Cat" if prediction[0][0] < 0.5 else "Dog"
    
    st.write(f"The uploaded image is a {class_label}.")

