import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Bike Predictor", page_icon="🏍️", layout="wide")

# Title
st.title("🏍️ Bike Rental Demand Predictor")
st.markdown("### 🚴 Predict bike demand using weather conditions")

# Load dataset
data = pd.read_csv("day.csv")

# -------------------------------
# 🧹 DATA CLEANING
# -------------------------------
data = data.drop_duplicates()
data = data.dropna()
data = data.reset_index(drop=True)

st.success("✅ Data cleaned successfully")

st.write("🔍 Missing values in each column:")
st.write(data.isnull().sum())

# Layout
col1, col2 = st.columns(2)

# -------------------------------
# 📂 DATA PREVIEW + DOWNLOAD
# -------------------------------
with col1:
    st.subheader("📂 Dataset Preview")
    st.dataframe(data.head())

    st.download_button(
        label="📥 Download Dataset",
        data=data.to_csv(index=False),
        file_name="bike_data.csv",
        mime="text/csv"
    )

# -------------------------------
# 🤖 MODEL TRAINING
# -------------------------------
X = data[['temp', 'hum', 'windspeed']]
y = data['cnt']

model = LinearRegression()
model.fit(X, y)

# -------------------------------
# 🌦️ USER INPUT
# -------------------------------
st.sidebar.header("🌦️ Enter Weather Conditions")

temp = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.3)
hum = st.sidebar.slider("💧 Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("🌬️ Wind Speed", 0.0, 1.0, 0.2)

season = st.sidebar.selectbox("🌸 Select Season", ["Spring", "Summer", "Fall", "Winter"])

# -------------------------------
# 🔮 PREDICTION
# -------------------------------
with col2:
    st.subheader("🔮 Prediction")

    if st.button("🚀 Predict Now"):
        prediction = model.predict([[temp, hum, windspeed]])
        value = int(prediction[0])

        st.success(f"🚴 Predicted Bike Rentals: {value}")

        # Metrics
        fraud = int(value * 0.1)
        real = value - fraud

        colA, colB, colC = st.columns(3)
        colA.metric("📊 Total Rentals", value)
        colB.metric("✅ Real Rentals", real)
        colC.metric("⚠️ Fraud/Noise", fraud)

        # -------------------------------
        # 📊 IMPROVED VISUALIZATION
        # -------------------------------
        st.subheader("📊 Input vs Prediction Analysis")

        # Input Features
        st.markdown("#### 🌦️ Input Weather Conditions")

        input_features = pd.DataFrame({
            'Features': ['Temperature', 'Humidity', 'Wind Speed'],
            'Values': [temp, hum, windspeed]
        })

        st.bar_chart(input_features.set_index('Features'))

        # Prediction Output
        st.markdown("#### 🚴 Predicted Bike Rentals")

        prediction_df = pd.DataFrame({
            'Category': ['Prediction'],
            'Predicted Rentals': [value]
        }).set_index('Category')

        st.bar_chart(prediction_df)

# -------------------------------
# 🛡️ SPAM DETECTION
# -------------------------------
st.subheader("🛡️ Rental Request Validation")

user_name = st.text_input("👤 Enter User Name")
booking_count = st.number_input("🚲 Number of Bikes Requested", 1, 50, 1)

if st.button("🔍 Validate Request"):
    if user_name.strip() == "":
        st.warning("⚠️ Name cannot be empty")
    elif booking_count > 10:
        st.error("🚨 Spam Alert: Too many bikes requested!")
    else:
        st.success("✅ Valid Request")

# -------------------------------
# 📈 INTERACTIVE GRAPHS
# -------------------------------
st.subheader("📈 Interactive Analysis")

# Line Chart
st.markdown("#### 📊 Bike Rental Trend")
st.line_chart(data['cnt'])

# Scatter Chart
st.markdown("#### 🌡️ Temperature vs Rentals")
st.scatter_chart(data[['temp', 'cnt']])

# Monthly Bar Chart
st.markdown("#### 📅 Monthly Average Rentals")
data['mnth'] = pd.to_datetime(data['dteday']).dt.month
monthly_avg = data.groupby('mnth')['cnt'].mean()
st.bar_chart(monthly_avg)

# -------------------------------
# 📊 METRICS
# -------------------------------
st.subheader("📊 Key Insights")

col5, col6, col7 = st.columns(3)
col5.metric("📌 Average Rentals", int(data['cnt'].mean()))
col6.metric("📈 Max Rentals", int(data['cnt'].max()))
col7.metric("📉 Min Rentals", int(data['cnt'].min()))

# Footer
st.markdown("---")
st.markdown("✨ Built with Streamlit | 🚀 ML Project with Spam Detection")
