import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Eco-Health Risk Predictor",
    layout="wide"
)

# 1. Introduction Dashboard
st.title("Eco-Health Risk Predictor Dashboard")
st.markdown("""
This dashboard presents the implementation of our machine learning
pipeline built in Jupyter Notebook.
It predicts regional public health risks based on environmental
metrics using a Random Forest Classifier.
""")

st.write("---")

# Data Loading
@st.cache_data
def load_data():
    return pd.read_csv("air_quality_health_impact_data.csv")

df = load_data()

# Smart Column Detection to Prevent Errors
available_columns = df.columns.tolist()

# 1. Identify the Correct Target Column
if 'HealthRiskClass' in available_columns:
    target_column = 'HealthRiskClass'
elif 'HealthRisk' in available_columns:
    target_column = 'HealthRisk'
elif 'Health_Risk' in available_columns:
    target_column = 'Health_Risk'
else:
    target_column = available_columns[-1]  # Use the last column if no target column is found

# 2. Create a List of Columns to Drop
columns_to_drop = [target_column]

if 'RecordID' in available_columns:
    columns_to_drop.append('RecordID')

# Layout Columns: Left for Data and Evaluation, Right for Interactive Sliders
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("1. Data Exploration and Model Evaluation")

    st.write("Dataset Sample View:")
    st.dataframe(df.head(4), use_container_width=True)

    # Train and Evaluate the Machine Learning Model
    X = df.drop(columns=columns_to_drop)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Plot the Confusion Matrix
    st.write("Confusion Matrix Heatmap:")

    cm = confusion_matrix(y_test, y_pred)

    # Generate Labels Based on Unique Target Values
    unique_labels = sorted(y.unique())
    class_names = [f"Risk Level {lbl}" for lbl in unique_labels]

    fig, ax = plt.subplots(figsize=(5, 3.8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    plt.ylabel('Actual Classes')
    plt.xlabel('Predicted Classes')
    st.pyplot(fig)

with col2:
    st.subheader("2. Interactive Prediction Engine")

    st.write(
        "Adjust the sliders below to see how the trained model "
        "predicts health risks in real time:"
    )

    user_inputs = {}

    # Automatically Generate Sliders for Each Feature
    for col in X.columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        step_val = 0.1 if max_val - min_val <= 100 else 1.0

        user_inputs[col] = st.slider(
            f"Current {col}",
            min_val,
            max_val,
            mean_val,
            step=step_val
        )

    # Generate a Prediction from the User Inputs
    user_df = pd.DataFrame([user_inputs])
    prediction = model.predict(user_df)[0]

    st.write("---")
    st.write("Model Prediction Result:")

    if prediction == 0:
        st.success("Low Health Risk Zone Predicted")
    elif prediction == 1:
        st.warning("Moderate Health Risk Zone Predicted")
    else:
        st.error("ALERT: High Health Risk Zone Detected")
