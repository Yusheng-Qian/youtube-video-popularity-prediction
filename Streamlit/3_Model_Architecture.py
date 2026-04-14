import streamlit as st

st.set_page_config(page_title="🤖 Model Architecture", layout="wide")

st.markdown("<h1 style='text-align: center;'>🤖 Model Architecture</h1>", unsafe_allow_html=True)

st.markdown("""
### 📌 Model Overview

This project uses **Linear Regression** to predict YouTube video views.

### 📊 Input Features

The model uses the following variables:

- Likes
- Comment count
- Title length
- Tag count
- Publish hour
- Publish month

### 🔢 Target Variable

The model predicts:

**Video Views**

### ⚙️ Training Pipeline

1. Data cleaning  
2. Feature engineering  
3. Train-test split  
4. Model training  
5. Model evaluation  

### 📈 Evaluation Metrics

We evaluate the model using:

- **R² Score**
- **Mean Absolute Error (MAE)**

These metrics help measure how well the model predicts video popularity.
""")
