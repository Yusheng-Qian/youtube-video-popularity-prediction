import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Dataset Visualization", layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 Dataset Visualization</h1>", unsafe_allow_html=True)

# 读取数据
df = pd.read_csv("data/dataset.csv")

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

st.subheader("📈 Views Distribution")

fig, ax = plt.subplots()
sns.histplot(df["views"], bins=50, ax=ax)
ax.set_title("Distribution of Video Views")

st.pyplot(fig)

st.subheader("❤️ Likes vs Views")

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="likes", y="views", ax=ax2)

st.pyplot(fig2)

st.subheader("💬 Comments vs Views")

fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x="comment_count", y="views", ax=ax3)

st.pyplot(fig3)
