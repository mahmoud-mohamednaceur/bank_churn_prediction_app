![2024-11-0618-30-54-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/85414dba-64e4-4239-98b4-464bfba6941b)

# Bank Churn Prediction Web App

In this project, I developed a **Bank Churn Prediction Web App** using Streamlit to predict the likelihood of a customer leaving a bank, also known as *customer churn*. This web application allows users to input customer details and instantly receive a churn prediction, making it a valuable tool for banks and financial institutions aiming to reduce customer turnover.

### Project Goals and Motivation

The main goal of this project was to create an easy-to-use, interactive application for bank staff or analysts to quickly assess churn risk based on key customer attributes. By providing actionable insights, the model helps banks take proactive measures to retain customers, potentially reducing churn rates and increasing customer satisfaction.

### Model Development

The prediction model for churn was built using supervised machine learning techniques, trained on a dataset of bank customers with information on demographic details, account balance, tenure, transaction activity, and more. Here’s a breakdown of the approach:

1. **Data Preprocessing**:
   - The dataset was cleaned and preprocessed to handle missing values, outliers, and categorical variables.
   - Key features were selected and scaled where necessary, as some machine learning algorithms perform better with standardized data.

2. **Feature Engineering**:
   - Based on domain knowledge, I created additional features that could enhance the model’s predictive power, such as the ratio of account balance to income or transaction frequency.
   - Categorical variables (like gender and geographical region) were encoded using techniques like one-hot encoding or label encoding to make them suitable for the model.

3. **Model Selection and Training**:
   - I experimented with various algorithms, including **Logistic Regression**, **Random Forest**, **Gradient Boosting**, and **XGBoost**.
   - After testing and evaluating each model using metrics like accuracy, precision, recall, and F1-score, I selected the model with the best performance, balancing accuracy and interpretability.
   - Hyperparameter tuning was conducted to optimize the model for higher accuracy and reduced overfitting.

4. **Evaluation**:
   - The chosen model was validated using a test set, and metrics like the AUC-ROC curve were used to evaluate its ability to differentiate between churned and non-churned customers effectively.
   - I also calculated feature importance scores to identify the most significant predictors of churn, such as tenure, account balance, and transaction activity, which provided additional insights for the app’s users.

### Building the Web App with Streamlit

**Streamlit** was chosen for the frontend because of its simplicity and flexibility, enabling rapid development of a clean and interactive interface. Key features of the app include:

- **User-Friendly Interface**: The app has a simple layout where users can enter customer attributes (such as age, account balance, credit score, etc.), and the model will generate a real-time churn prediction.
- **Interactive Input Fields**: The app uses Streamlit’s interactive widgets (like sliders, drop-down menus, and number inputs) to make it easy for users to input data.
- **Real-Time Predictions**: Once the user inputs all necessary data, the app processes the inputs and displays a churn prediction almost instantly, providing a seamless experience.




