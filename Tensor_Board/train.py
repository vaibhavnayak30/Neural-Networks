import torch
import torch.nn as nn 
import torch.functional as F

# Import tensorboard summary writer 
from torch.utils.tensorboard import SummaryWriter

# Create summary writer instance 
writer = SummaryWriter()

# Generate a range of values and convert it to single column tensor 
x = torch.arange(-5, 5, 0.1).view(-1,1)
y = -5 * x + 0.1 * torch.randn(x.size())

# Generate a single input and single output 
model = torch.nn.Linear(1,1)
criterian = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterian(y1, y)
        writer.add_scalar(tag='Loss/train', scalar_value=loss, global_step=epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(10)
writer.flush()
writer.close()



