import pandas as pd
import numpy as np

# Student data: hours studied and attendance
data = {
    'Hours_Study': [2, 4, 6, 8],
    'Attendance': [60, 70, 80, 90]
}
df = pd.DataFrame(data)
X = df.values

# Define weights and bias
weights = np.array([0.5, 0.3])
bias = 0.2

# Forward propagation: dot product of inputs and weights, plus bias
output = np.dot(X, weights) + bias
print("Output Predictions:")
print(output)

# Backward propagation: compute gradients of the weights
loss_gradient = np.ones(len(X))
grad_weights = np.dot(X.T, loss_gradient)
print("Gradient of Weights:")
print("Hours_Study:", grad_weights[0])
print("Attendance:", grad_weights[1])
