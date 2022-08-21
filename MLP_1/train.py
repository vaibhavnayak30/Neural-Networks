# import all necessary libraries
from NN import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch


def next_batch(inputs, targets, batchSize):
    # loop over the dataset
    for i in range(0, inputs.shape[0], batchSize):
        # yield a tuple of current batched data and labels
        yield inputs[i:i + batchSize], targets[i:i + batchSize]


# specify our batch_size, number of epochs and learning rate
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2

# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using: {}...".format(DEVICE))

# generate 3-class classification problem with 1000 data points
print("[INFO] preparing dataset...")
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3, cluster_std=2.5, random_state=95)

# create training and testing splits and convert to pytorch tensors
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.15, random_state=95)
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

''' flash model to device '''
mlp = mlp.get_training_model().to(DEVICE)
print(mlp)

# initialize optimizer and loss function
opt = SGD(mlp.parameters(), lr=LR)
lossFunc = nn.CrossEntropyLoss()

# Create a template to summarize current training progress
trainTemplate = "epoch: {} test loss: {:.3f} test accuracy {:.3f}"

# loop through the epochs
for epoch in range(0, EPOCHS):
    # initialize tracker variable and set out model to trainable
    print("[INFO] epoch: {}...".format(epoch + 1))
    trainLoss = 0
    trainAcc = 0
    samples = 0
    mlp.train()

    # loop over current batch of data
    for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
        # Flash data to device and run it through model to calculate loss
        (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
        predictions = mlp(batchX)
        loss = lossFunc(predictions, batchY.long())

        ''' zero the gradients accumulated in previous steps,
        perform backpropagation, and update the model parameters'''
        opt.zero_grad()
        loss.backward()
        opt.step()

        # update training loss, accuracy and number of samples visited
        trainLoss += loss.item() * batchY.size(0)
        trainAcc += (predictions.max(1)[1] == batchY).sum().item()
        samples += batchY.size(0)

        # display model progress on current training batch
        trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
        print(trainTemplate.format(epoch + 1, (trainLoss / samples), (trainAcc / samples)))

    # initialize tracker variables for testing, the set model to eval mode
    testLoss = 0
    testAcc = 0
    samples = 0
    mlp.eval()

    # initialize no gradient context
    with torch.no_grad():
        # loop over current batch of data
        for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):
            # Flash data to device
            (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

            ''' run data through device and calculate loss'''
            predictions = mlp(batchX)
            loss = lossFunc(predictions, batchY.long())

            # update loss, accuracy and number of samples
            testLoss += loss.item() * batchY.size(0)
            testAcc += (predictions.max(1)[1] == batchY).sum().item()
            samples += batchY.size(0)

        # display model progress on the current test data
        testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
        print(testTemplate.format(epoch + 1, (testLoss / samples), (testAcc / samples)))
        print("")


