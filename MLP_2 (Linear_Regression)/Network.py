# Linear Regression Using Pytorch
"""
Steps:
1. Design the model
2. Construct loss and optimizer
3. Training loop
    - forward pass: compute prediction and loss
    - backward pass: gradients
    - update weights
"""

# import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Prepare the data
X_numpy , y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# Convert numpy array to pytorch tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples, n_features = X.shape
input_size = n_features
output_size = 1
learning_rate = 0.01

# Design the model
model = nn.Linear(input_size, output_size)

# Define the loss and optimizer
criteria = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# Training loop
num_epoch = 100

for epoch in range(num_epoch):
    # forward pass
    y_predicted = model(X)
    loss = criteria(y_predicted, y)

    # empty the gradient
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1} , loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,predicted, 'b')
plt.show()