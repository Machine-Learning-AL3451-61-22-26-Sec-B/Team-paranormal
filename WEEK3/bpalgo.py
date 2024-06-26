import streamlit as st
import numpy as np

# Neural Network class
class Neural_Network(object):
    def __init__(self):
        # Parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        # Weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        # Forward propagation through our network
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)

    def backward(self, X, y, o):
        # Backward propagate through the network
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

# Initialize the neural network
NN = Neural_Network()

# Streamlit app
st.title('22AIB TEAM PARANORMAL')
st.title('Simple Neural Network with Streamlit')

# Input data
st.write('Input data (hours sleeping, hours studying):')
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)     
y = np.array(([92], [86], [89]), dtype=float)           

# Scale units
X = X / np.amax(X, axis=0)
y = y / 100

st.write(f"Input: \n{X}")
st.write(f"Actual Output: \n{y}")

# Train the network
if st.button('Train the Neural Network'):
    NN.train(X, y)
    predicted_output = NN.forward(X)
    loss = np.mean(np.square(y - predicted_output))
    
    st.write(f"Predicted Output: \n{predicted_output}")
    st.write(f"Loss: \n{loss}")
