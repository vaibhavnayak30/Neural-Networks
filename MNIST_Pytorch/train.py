# import all necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# define model architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.cn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.dp1 = nn.Dropout2d(0.1)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12x12x32
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op


# Define training and inference routine
def train(model, device, train_dataloader, optim, epoch):
    model.train()
    for b_i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        pred_prob = model(X)
        loss = F.nll_loss(pred_prob, y)        # nll is negative log loss
        loss.backward()
        optim.step()
        if b_i % 10 == 0:
            print('epoch: {} [{}/{} ({:.0f}%)]\t training_loss: {:.6f}'.format(
                epoch, b_i * len(X), len(train_dataloader.dataset),
                100. * b_i / len(train_dataloader), loss.item()))


def test(model, device, test_dataloader):
    model.eval()
    loss = 0
    success = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss += F.nll_loss(pred_prob, y, reduction='sum').item()        # loss summed across batches
            pred = pred_prob.argmax(dim = 1, keepdims=True)
            success += pred.eq(y.view_as(pred)).sum().item()

    loss /= len(test_dataloader.dataset)

    print('\nTest Dataset: Overall Loss: {:.4f}, Overall Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss,success,len(test_dataloader.dataset),
        100 * success / len(test_dataloader.dataset)))


# Create data loaders
train_dataloader = torch.utils.data.DataLoader(
    dataset=datasets.MNIST('<path_to_folder_for_saving>', train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.1302,), (0.3069,))])),
    batch_size=500, shuffle=False)

test_dataloader = torch.utils.data.DataLoader(
    dataset=datasets.MNIST('<path_to_folder_for_saving>', train=False, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.1302,), (0.3069,))])),
    batch_size=500, shuffle=False)


# Define optimiser
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet()
model.to(device)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)

# Model training
for epoch in range(1,3):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)


# Run inference on trained model
test_samples = enumerate(test_dataloader)
b_i, (sample_data, sample_targets) = next(test_samples)

# Plot the sample
plt.imshow(sample_data[0][0], cmap='gray', interpolation=None)

# Inference on test image using trained model
print(f'Model prediction is: {model(sample_data).data.max(1)[1][0]}')
print(f'Ground truth is: {sample_target[0]}')

# Another method for inferencing
# print(f'Model prediction is: {torch.argmax(model(sample_data).data,1)[0]}')
# print(f'Ground truth is: {sample_target[0]}')