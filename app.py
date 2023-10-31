import streamlit as st
import tensorflow as tf
import graphviz
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, LSTM
import numpy as np

# Define a function to create a neural network graph for Fully Connected Neural Network (FCNN)
def create_fcnn(num_layers, num_neurons, activation_function):
    inputs = Input(shape=(2,))
    x = inputs
    for _ in range(num_layers):
        x = Dense(num_neurons, activation=activation_function)(x)
    model = Model(inputs=inputs, outputs=x)
    return model

# Define a function to create a neural network graph for LeNet
def create_lenet():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(6, (5, 5), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define a function to create a neural network graph for Convolutional Neural Network (CNN)
def create_cnn(kernel_size, pooling_size):
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (kernel_size, kernel_size), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(pooling_size, pooling_size))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define a function to create a neural network graph for Recurrent Neural Network (RNN)
def create_rnn(num_time_steps):
    inputs = Input(shape=(num_time_steps, 64))
    x = LSTM(32, return_sequences=True)(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Cache the functions to prevent recomputation
@st.cache_resource
def get_model(network_type, num_layers, num_neurons, activation_function, kernel_size, pooling_size, num_time_steps):
    if network_type == "Fully Connected Neural Network (FCNN)":
        return create_fcnn(num_layers, num_neurons, activation_function)
    elif network_type == "LeNet":
        return create_lenet()
    elif network_type == "Convolutional Neural Network (CNN)":
        return create_cnn(kernel_size, pooling_size)
    elif network_type == "Recurrent Neural Network (RNN)":
        return create_rnn(num_time_steps)

# Streamlit app
st.title("Neural Network Visualization App")

# User input
network_type = st.selectbox("Select Network Type", ["Fully Connected Neural Network (FCNN)", "LeNet", "Convolutional Neural Network (CNN)", "Recurrent Neural Network (RNN)"])

# Display relevant input fields based on the selected network type
if network_type in ["Fully Connected Neural Network (FCNN)", "LeNet"]:
    num_layers = st.slider("Number of Layers", 1, 10, 3)
    num_neurons = st.slider("Number of Neurons per Layer", 1, 256, 64)
    activation_function = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
    kernel_size = None
    pooling_size = None
    num_time_steps = None
elif network_type == "Convolutional Neural Network (CNN)":
    kernel_size = st.slider("Kernel Size", 3, 7, 5)
    pooling_size = st.slider("Pooling Size", 2, 5, 2)
    num_layers = num_neurons = activation_function = num_time_steps = None
elif network_type == "Recurrent Neural Network (RNN)":
    num_time_steps = st.slider("Number of Time Steps", 1, 50, 10)
    num_layers = num_neurons = activation_function = kernel_size = pooling_size = None

# Create or retrieve the neural network based on the selected type
model = get_model(network_type, num_layers, num_neurons, activation_function, kernel_size, pooling_size, num_time_steps)

# Display network architecture using Graphviz
st.graphviz_chart(tf.keras.utils.model_to_dot(model, show_shapes=True, expand_nested=True))
