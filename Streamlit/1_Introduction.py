import streamlit as st

# 设置页面标题和图标
st.set_page_config(page_title="📘 01 Introduction", layout="wide")

# 页面主标题
st.markdown("""
    <h1 style='text-align: center;'>📘 01 Introduction</h1>
""", unsafe_allow_html=True)

# 项目简介
st.markdown("""
### 🎯 Project Goal
This app aims to help users analyze YouTube video trends and predict their popularity using a linear regression model.

### ℹ️ Why It Matters
- YouTube is one of the largest video platforms with billions of views daily.
- Understanding video performance is valuable for creators, marketers, and researchers.
- Predictive modeling can offer data-driven insights into what makes content popular.

### ⚙️ Method Overview
We apply **linear regression** to predict video views using variables such as:
- Title length
- Likes
- Comments
- Duration

The model is trained on a public dataset of real YouTube videos.

### 📚 Use Cases
- 🎥 Creators: Optimize video titles and tags to increase visibility.
- 📈 Marketers: Analyze content trends to guide strategy.
- 🧠 Students: Understand how machine learning works in media analysis.

### 🛠 Technologies Used
- Python 🐍
- Streamlit 📊
- Pandas, Seaborn, scikit-learn

---
📌 Use the left sidebar to explore the dataset, understand the model, and try making your own predictions.
""")
