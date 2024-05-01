import streamlit as st
import pandas as pd
import numpy as np

def train_model(df):
    X = df.drop(columns=['Embarked'])
    y = df['Embarked']
    # Dummy model for demonstration
    class_counts = y.value_counts().to_dict()
    most_common_class = max(class_counts, key=class_counts.get)
    return most_common_class

def evaluate_model(df, most_common_class):
    y_true = df['Embarked']
    y_pred = np.full_like(y_true, most_common_class)
    accuracy = np.mean(y_true == y_pred)
    return accuracy, y_pred

def main():
    st.title("Dummy Classifier")
    st.write("This app demonstrates a dummy classifier using your own data.")

    # Upload user data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display dataset
        st.subheader("Uploaded Dataset")
        st.write(df)

        # Train model
        most_common_class = train_model(df)

        # Display most common class
        st.subheader("Most Common Class")
        st.write("The most common class in the dataset is:", most_common_class)

        # Evaluate model
        accuracy, y_pred = evaluate_model(df, most_common_class)
        st.subheader("Model Evaluation")
        st.write("Accuracy:", accuracy)

        # Display predicted values
        st.subheader("Predicted Values")
        st.write("Predicted values:", y_pred)

if __name__ == "__main__":
    main()
