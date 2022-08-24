"""
Logistic Regression using pytorch
1. Dataset Preparation
2. Design the model
3. Construct loss and optimizer
4. Training loop
    - Forward pass: compute prediction and loss
    - Backward pass: compute gradients
    - Update weights
"""

# import necessary libraries
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

# Divide data into training and testing chunks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# scale our features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Covert data to torch tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape y
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)


# Design model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=n_input_features, out_features=1)

    def forward(self, x):
        model = torch.sigmoid(self.linear(x))
        return model


# Initialize the model
lr_model = LogisticRegression(n_features)

# Model hyperparameters
training_epochs = 100
lr = 0.01

# Initiate loss and optimizer function
criteria = torch.nn.BCELoss()
optimiser = torch.optim.SGD(lr_model.parameters(), lr)

# Load model and tensor on available device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {} for processing".format(DEVICE))
lr_model.to(DEVICE)
X_train = X_train.to(DEVICE)
y_train = y_train.to(DEVICE)

# Print format
out_format = "epoch: {}, loss: {:.4f}"

# Training Loop
for epoch in range(training_epochs):
    # forward pass
    y_predicted = lr_model.forward(X_train)

    # loss calculation
    loss = criteria(y_predicted, y_train)

    # remove all previous gradients
    optimiser.zero_grad()

    # calculate gradients for backpropagation
    loss.backward()

    # weights update using optimizer
    optimiser.step()

    if (epoch+1) % 10 == 0:
        print(out_format.format(epoch+1, loss.item()))

# Load evaluation tensor to device
X_test = X_test.to(DEVICE)
y_test = y_test.to(DEVICE)

# Evaluation loop
with torch.no_grad():
    # forward pass
    y_predicted = lr_model(X_test)
    y_predicted_classes = y_predicted.round()

    # calculate accuracy
    acc = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])

    print("Accuracy: {:.4f}".format(acc))



