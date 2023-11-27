#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


# In[2]:


# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# In[3]:


# Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[4]:


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[5]:


# Train the model
model.fit(train_images, train_labels, epochs=5)


# In[8]:


# Streamlit App
st.title("Simple Neural Network Classifier")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image
    img_array = tf.image.decode_image(uploaded_file.read(), channels=1)
    img_array = tf.image.resize(img_array, [28, 28])
    img_array = tf.image.rgb_to_grayscale(img_array)
    img_array = tf.reshape(img_array, (1, 28, 28)) / 255.0

    # Make prediction using the trained model
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # Display the uploaded image
    st.image(img_array[0], caption="Uploaded Image", use_column_width=True)

    # Display the prediction
    st.write(f"Prediction: {predicted_label}")


# In[ ]:




