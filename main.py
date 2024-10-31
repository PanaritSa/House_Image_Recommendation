import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Load feature list and filenames
try:
    feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
    filenames = pickle.load(open("filenames.pkl", "rb"))
    st.write("Loaded feature vector and filenames successfully.")
except Exception as e:
    st.write(f"Error loading files: {e}")

# Initialize model
try:
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model.trainable = False
    model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])
    model.summary()
    st.write("Model initialized successfully.")
except Exception as e:
    st.write(f"Error initializing model: {e}")

st.title('House Recommender System')

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        # Create uploads directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.write(f"Error saving uploaded file: {e}")
        return None

# Function to extract features
def extract_feature(img_path, model):
    try:
        img = cv2.imread(img_path)
        if img is None:
            st.write("Error: Could not read the image with OpenCV.")
            return None
        img = cv2.resize(img, (224, 224))
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)
        return normalized
    except Exception as e:
        st.write(f"Error extracting features: {e}")
        return None

# Recommendation function
def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        _, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.write(f"Error in recommendation: {e}")
        return None

# Function to get folder name from file path
def get_folder_name(file_path):
    return os.path.basename(os.path.dirname(file_path))

# File upload
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img, caption="Uploaded Image")

        # Extract features and recommend
        features = extract_feature(file_path, model)
        if features is not None:
            indices = recommend(features, feature_list)
            if indices is not None:
                # Show recommended images with folder names only
                col1, col2, col3, col4, col5 = st.columns(5)
                try:
                    for i, col in enumerate([col1, col2, col3, col4, col5]):
                        image_path = filenames[indices[0][i]]
                        folder_name = get_folder_name(image_path)
                        with col:
                            st.image(image_path, caption=folder_name)
                except Exception as e:
                    st.write(f"Error displaying recommended images: {e}")
            else:
                st.write("No recommendations available.")
    else:
        st.header("Error occurred during file upload.")
