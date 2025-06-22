import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained assets
model = tf.keras.models.load_model("churn_model_tf.keras", compile=False)
columns = joblib.load("columns.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("🔍 Customer Churn Predictor")
st.markdown("Enter customer details on the left sidebar to predict churn probability.")

# Sidebar input fields
st.sidebar.header("📋 Input Features")
user_input = {}
for col in columns:
    user_input[col] = st.sidebar.number_input(f"{col}", value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input], columns=columns)

# Show input preview
st.subheader("🧾 Input Preview")
st.write(input_df)

input_df = input_df[columns]  # forces correct column order and count
input_np = input_df.to_numpy().astype(np.float32)
scaled_input = scaler.transform(input_np)
st.write("🧪 Final shape for prediction:", scaled_input.shape)


# Predict on button click
if st.button("🔮 Predict Churn"):
    st.write("🧠 Scaled input type:", type(scaled_input))
    st.write("📐 Scaled input shape:", scaled_input.shape)


    try:
        input_df = input_df[columns]
        input_np = input_df.to_numpy().astype(np.float32)
        st.write("✅ Input converted to NumPy")

        scaled_input = scaler.transform(input_np)
        st.write("✅ Scaled Input:", scaled_input)

        prediction = model.predict(scaled_input)
        st.write("🎯 Model Raw Output:", prediction)

        prob = prediction[0][0]
        st.success(f"**Churn Probability:** `{prob:.2%}`")

        if prob > 0.5:
            st.error("⚠️ Likely to Churn")
        else:
            st.success("✅ Likely to Stay")
    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))

