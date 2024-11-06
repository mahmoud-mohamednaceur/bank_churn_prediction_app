"""
Machine Learning Prediction App

Purpose:
This code builds an interactive web application using Streamlit that allows users to input feature values and receive real-time predictions from a pre-trained machine learning model.
The model is loaded from a serialized file (.pkl), ensuring quick access and consistent predictions without the need for retraining.
The app is designed to be user-friendly, with a sidebar for input features and a main section to display the prediction results.

Key Features:
- Interactive input sliders in the sidebar to collect user-provided feature values.
- A simple and responsive layout that displays predictions in real-time.
- Loaded model allows efficient deployment without training from scratch.
- Suitable for use cases like regression or classification tasks where predictions are generated based on user inputs.

"""


import streamlit as st
import pandas as pd
import pickle
from lib.utilities.help_functions import HelperFunctions

# Load the trained model
with open('trained_models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Sidebar Configuration
st.sidebar.image("data:image/jpeg;base64,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")
st.image("https://user-images.githubusercontent.com/58620359/174948746-5dc3418a-8296-4cc8-9561-f8f12ca9a0a4.png")
st.sidebar.title("Customer Exit Prediction Tool")
st.sidebar.markdown(
    "Use this tool to predict the likelihood of customers leaving the bank. "
    "Upload a test dataset, preprocess it, and get predictions instantly."
)

# Main Title
st.title('Bank Customer Churn Prediction')
st.write("""
    This application uses a machine learning model to predict the probability of customer churn.
    The predictions can help in identifying customers at risk of leaving.
""")

# Load and preprocess the test dataset
st.header("Step 1: Load and Preprocess Test Data")
test_dataset = pd.read_csv("datasets/test.csv")

# Show raw data if the user requests it
if st.checkbox("Show Raw Test Data"):
    st.write(test_dataset.head())

# Data Preprocessing
st.write("Preprocessing the dataset for prediction...")
processed_input = HelperFunctions().scale_dataframe(
    HelperFunctions().encode_features(
        test_dataset.copy().drop(columns=["CustomerId", "Surname"]),
        global_encoding_method="one-hot"
    ),
    method='minmax'
)
st.success("Data preprocessing completed!")

# Button to make predictions
st.header("Step 2: Predict Customer Churn")
if st.button('Run Prediction'):
    # Generate prediction probabilities
    predictions = model.predict_proba(processed_input)

    # Create a DataFrame for results
    results_df = pd.DataFrame(predictions, columns=["Retention Probability", "Churn Probability"])

    # Display results
    st.subheader("Prediction Results")
    st.write("Here are the churn probabilities for each customer in the test dataset:")
    st.dataframe(results_df)

    # Highlight customers likely to churn
    high_churn = results_df[results_df["Churn Probability"] > 0.5]
    if not high_churn.empty:
        st.warning("Customers with a high likelihood of churning:")
        st.dataframe(high_churn)
    else:
        st.info("No customers with high churn probability detected.")
else:
    st.info("Click the button above to run the prediction model.")
