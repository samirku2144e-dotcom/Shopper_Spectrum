import os
import joblib
import streamlit as st
import numpy as np

st.set_page_config(page_title="SHOPPER SPECTRUM", page_icon="🛒")

st.title("CUSTOMER SEGMENTATION")

# 1. Load the pre-trained models safely
@st.cache_resource
def load_models():
    # Get the current folder (pages) and go up one level to the main root folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    # Safely construct the exact paths to the models
    scaler_path = os.path.join(parent_dir, 'rfm_scaler.pkl')
    kmeans_path = os.path.join(parent_dir, 'kmeans_model.pkl')
    
    try:
        loaded_scaler = joblib.load(scaler_path)
        loaded_kmeans = joblib.load(kmeans_path)
        return loaded_scaler, loaded_kmeans
    except FileNotFoundError:
        st.error(f"Model files not found! Looked in:\n{scaler_path}\n{kmeans_path}")
        return None, None

# THIS LINE IS CRITICAL: It assigns the loaded models to the variables 'scaler' and 'kmeans'
scaler, kmeans = load_models()

# 2. Create the input form
recency = st.number_input("Recency (days since last purchase)", min_value=0, value=325)
frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=1)
monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=765322.00)

# 3. Prediction Logic
if st.button("PREDICT SEGMENT"):
    # Check to make sure the models actually loaded before trying to use them
    if kmeans is not None and scaler is not None:
        
        # Convert inputs into the shape the model expects: 2D array
        input_features = np.array([[recency, frequency, monetary]])
        
        # Scale the data using the fitted scaler
        scaled_features = scaler.transform(input_features)
        
        # Predict the cluster
        cluster_prediction = kmeans.predict(scaled_features)[0]
        
        # Map the cluster number to a human-readable label 
        # (Adjust these string labels if your notebook results differ slightly)
        cluster_mapping = {
            0: "At-Risk",
            1: "Regular",
            2: "High-Value",
            3: "Occasional Shopper"
        }
        
        segment_label = cluster_mapping.get(cluster_prediction, f"Cluster {cluster_prediction}")
        
        st.success(f"This customer belongs to: **{segment_label}**")
    else:
        st.error("Cannot predict because the models failed to load. Please check your .pkl files.")