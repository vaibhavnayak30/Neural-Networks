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


