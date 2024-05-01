import streamlit as st
import numpy as np

# Function to normalize data
def normalize(data):
    return data / np.amax(data, axis=0)

# Function to define sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to define gradient of sigmoid
def sigmoid_grad(x):
    return x * (1 - x)

# Streamlit app
def main():
    st.title("Neural Network Prediction")

    # Input form for user to enter numbers
    st.subheader("Enter Numbers")
    number1 = st.number_input("Enter first number", value=2.0)
    number2 = st.number_input("Enter second number", value=9.0)
    number3 = st.number_input("Enter third number", value=1.0)
    number4 = st.number_input("Enter fourth number", value=5.0)
    number5 = st.number_input("Enter fifth number", value=3.0)
    number6 = st.number_input("Enter sixth number", value=6.0)

    if st.button("Predict"):
        # Convert input data to numpy array and normalize
        X = np.array([[number1, number2], [number3, number4], [number5, number6]])
        X_normalized = normalize(X)

        # Neural network parameters
        input_neurons = 2
        hidden_neurons = 3
        output_neurons = 1

        # Initialize weights and biases
        wh = np.random.uniform(size=(input_neurons, hidden_neurons))    # 2x3
        bh = np.random.uniform(size=(1, hidden_neurons))    # 1x3
        wout = np.random.uniform(size=(hidden_neurons, output_neurons))    # 3x1
        bout = np.random.uniform(size=(1, output_neurons))

        # Feedforward
        h_ip = np.dot(X_normalized, wh) + bh
        h_act = sigmoid(h_ip)
        o_ip = np.dot(h_act, wout) + bout
        output = sigmoid(o_ip)

        st.subheader("Prediction:")
        st.write(output)

if __name__ == "__main__":
    main()
