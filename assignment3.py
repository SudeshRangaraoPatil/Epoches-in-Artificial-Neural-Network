#!/usr/bin/env python
# coding: utf-8

# 

# In[11]:


import numpy as np


# In[12]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[13]:


def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.uniform(-1, 1, (hidden_size, input_size))
    b1 = np.random.uniform(-1, 1, (hidden_size, 1))
    W2 = np.random.uniform(-1, 1, (output_size, hidden_size))
    b2 = np.random.uniform(-1, 1, (output_size, 1))
    return W1, b1, W2, b2


# In[14]:


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A1, A2


# In[15]:


def compute_loss(Y, A2):
    m = Y.shape[1]
    loss = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss


# In[16]:


def backward_propagation(X, Y, A1, A2, W1, W2, b1, b2):
    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


# In[17]:


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


# In[18]:


def train(X, Y, hidden_size, output_size, learning_rate, num_epochs):
    input_size = X.shape[0]
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(num_epochs):
        A1, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(Y, A2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W1, W2, b1, b2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return W1, b1, W2, b2


# In[19]:


def predict(X, W1, b1, W2, b2):
    _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.round(A2)
    return predictions


# In[20]:


# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.array([[0, 1, 1, 0]])

hidden_size = 2
output_size = 1
learning_rate = 0.01
num_epochs = 10000

W1, b1, W2, b2 = train(X, Y, hidden_size, output_size, learning_rate, num_epochs)

# Make predictions
predictions = predict(X, W1, b1, W2, b2)
print("Predictions:", predictions)

