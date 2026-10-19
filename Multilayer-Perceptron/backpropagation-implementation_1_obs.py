"""
A simple neural network with 2 input neurons, 2 hidden neurons (ReLU),
and 1 output neuron (sigmoid) trained using Binary Cross-Entropy (BCE).
"""


import numpy as np

# One training example
x = np.array([[1.0, 2.0]])  # shape (1, 2)
y = np.array([[1.0]])       # shape (1, 1)

# Initialize parameters
W1 = np.array([
    [0.5, -0.2],
    [0.3,  0.8]
])
b1 = np.array([[0.1, -0.1]])

W2 = np.array([[0.4, -0.5]])
b2 = np.array([[0.2]])

learning_rate = 0.1
eps = 1e-12


def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    # Numerically stable for this small example
    return 1.0 / (1.0 + np.exp(-z))


# 1. FORWARD PROPAGATION
z1 = x @ W1.T + b1
h = relu(z1)

z2 = h @ W2.T + b2
y_pred = sigmoid(z2)

# 2. LOSS: binary cross-entropy
p = np.clip(y_pred, eps, 1 - eps)
loss = -np.mean(
    y * np.log(p) + (1 - y) * np.log(1 - p)
)

# 3. BACKPROPAGATION
# Sigmoid + BCE, with mean over the batch
batch_size = x.shape[0]
delta2 = (y_pred - y) / batch_size

dW2 = delta2.T @ h
db2 = np.sum(delta2, axis=0, keepdims=True)

delta1 = (delta2 @ W2) * (z1 > 0)

dW1 = delta1.T @ x
db1 = np.sum(delta1, axis=0, keepdims=True)

# 4. GRADIENT DESCENT UPDATE
W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1

W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2

print("Prediction before update:", y_pred)
print("Loss before update:", loss)
print("Updated W1:\n", W1)
print("Updated b1:\n", b1)
print("Updated W2:\n", W2)
print("Updated b2:\n", b2)

# 5. FORWARD PASS AGAIN WITH UPDATED PARAMETERS
z1_new = x @ W1.T + b1
h_new = relu(z1_new)
z2_new = h_new @ W2.T + b2
y_pred_new = sigmoid(z2_new)

p_new = np.clip(y_pred_new, eps, 1 - eps)
loss_new = -np.mean(
    y * np.log(p_new) + (1 - y) * np.log(1 - p_new)
)

print("Prediction after update:", y_pred_new)
print("Loss after update:", loss_new)
