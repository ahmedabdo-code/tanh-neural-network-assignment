import numpy as np

# tanh activation
def tanh(x):
    return np.tanh(x)

# random weights [-0.5, 0.5]
np.random.seed(0)

W1 = np.random.uniform(-0.5, 0.5, (2, 2))
W2 = np.random.uniform(-0.5, 0.5, (2, 1))

b1 = 0.5
b2 = 0.7

# input example
X = np.array([[0.6, 0.1]])

# forward pass
z1 = np.dot(X, W1) + b1
a1 = tanh(z1)

z2 = np.dot(a1, W2) + b2
output = tanh(z2)

print("Output:", output)