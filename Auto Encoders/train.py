# import all necessary modules
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

# Prepare dataset
# Convert images to pytorch tensors
tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data',
                         train=True,
                         download=True,
                         transform=tensor_transform)

# Dataloader is used for loading the data for training
loader = DataLoader(dataset=dataset,
                    batch_size=32,
                    shuffle=True)


# Network
class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()

        # Encoder layer with Linear layer and Relu activation function
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        # Decoder layer with Linear layer and Relu activation function and sigmoid
        # in the end layer to output values between 0 nd 1
        self.decoder = (torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 784),
            torch.nn.Sigmoid()
        ))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Model initialization
model = auto_encoder()

# Initialize loss and optimizer function
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

# Hyperparameters for model training
training_epochs = 20
losses = list()
outputs = list()

# Training loop
for epoch in range(training_epochs):
    for image, _ in loader:
        # reshape the image
        image = image.reshape(-1, 784)  # 28x28

        # Output of encoder
        reconstructed = model(image)

        # Calculate the loss
        loss = loss_function(reconstructed, image)

        # set gradient from previous runs to zero
        optimizer.zero_grad()

        # compute the gradients for backpropagation
        loss.backward()

        # update the weights
        optimizer.step()

        # storing the loss in a list for plotting
        losses.append(loss.item())

    template = 'epochs:{}, loss:{}'
    print(template.format(epoch, np.mean(losses)))

# Define the plot style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plot
plt.plot(losses[-100:])
plt.show()
