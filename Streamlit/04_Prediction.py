import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="📈 Prediction", layout="wide")

st.markdown("<h1 style='text-align: center;'>📈 Video View Prediction</h1>", unsafe_allow_html=True)

df = pd.read_csv("data/dataset.csv")

df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
df["publish_month"] = df["publish_time"].dt.month
df = df.dropna()

features = ["likes", "comment_count", "title_length", "tag_count", "publish_hour", "publish_month"]

X = df[features]
y = df["views"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("🔮 Predict Views for a New Video")

likes = st.number_input("Likes", 0, 1000000, 50000)
comments = st.number_input("Comments", 0, 500000, 10000)
title_length = st.slider("Title Length", 5, 100, 40)
tag_count = st.slider("Tag Count", 0, 30, 10)
publish_hour = st.slider("Publish Hour", 0, 23, 17)
publish_month = st.selectbox("Publish Month", list(range(1,13)))

input_data = pd.DataFrame([{
    "likes": likes,
    "comment_count": comments,
    "title_length": title_length,
    "tag_count": tag_count,
    "publish_hour": publish_hour,
    "publish_month": publish_month
}])

prediction = model.predict(input_data)[0]

st.success(f"Predicted Views: {int(prediction):,}")
