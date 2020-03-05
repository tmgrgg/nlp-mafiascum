# -*- coding: utf-8 -*-
"""Copy of NLP Mafia (Logistic Regression).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d0X3sqMr44SxIj7ub0bqsayWuDrNs9j4

# NLP Mafia Project (Logistic Regression Baseline Model)

## Clone into the repository to obtain the data
"""

!git clone https://bitbucket.org/bopjesvla/thesis.git

!cp thesis/src/* .

"""## Import the required modules"""

import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.nn import functional as F

docs = pd.read_pickle('docs.pkl', compression='gzip')
docs

"""## Ritwick's Code (Testing)"""

# Reset the indices as some indices might be missing or have null values
docs = docs.reset_index()
docs

example = MyDataset(docs.values, docs['scum'].values)

# All Game IDs are being printed, which is good!
for x, y in example:
    print(x[2])

# Creating a random train/test split of the dataset
train_size = int(0.8 * len(example))
test_size = len(example) - train_size

train_dataset_pytorch, test_dataset_pytorch = torch.utils.data.random_split(example, [train_size, test_size])
print(type(train_dataset_pytorch))

# Parameters
params = {
    'batch_size' : 50,
    'shuffle' : True,
    'num_workers' : 6
}
max_epochs = 100


#training_set = Dataset(train_dataset_pytorch, test_dataset_pytorch)
training_generator = DataLoader(train_dataset_pytorch, **params)

for x, y in training_generator:
    print(x)

# 80/20 Train-Test split of the dataset

train_dataset, test_dataset = train_test_split(docs, test_size = 0.2)

"""## Create the Model Class"""

# Logistic Regression module

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

"""## Instantiate the Model, Loss and Optimizer Classes"""

# Splitting the training and test datasets in order to store the 
# input and labels separately

train_labels = train_dataset.scum.values.astype(np.float64)
train_data   = train_dataset.vector_FastText_wiki.values

test_labels = test_dataset.scum.values.astype(np.float64)
test_data   = test_dataset.vector_FastText_wiki.values

# Code snippet that Tom provided. Converts the input and
# labels to Torch Tensors.

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        
        return X, y
        
    def __len__(self):
        return len(self.y)

train = MyDataset(train_data, train_labels)
test  = MyDataset(test_data, test_labels)

train_length = train.__len__()
test_length  = test.__len__()

batch_size = 50
iters = int(np.floor(train_length / batch_size) + 1)
epochs = 100
input_dim = 300
output_dim = 2
learning_rate = 0.01

model     = LogisticRegression(input_dim, output_dim)
model.double()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

iter = 0

for epoch in range(epochs):
  batch_start = 0
  batch_end   = batch_size
  to_end = 0
  last_loop = False
  for k in range(1, iters+1):
    for i in range(batch_start, batch_end):
      text    = train[i][0]
      labels  = train[i][1].view(1, )
      outputs = model(text).view(1, -1)
      loss    = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    if k % 10 == 0:
      # calculate Accuracy
      correct = 0
      total   = 0
      for j in range(test_length):
        test_text   = test[j][0]
        test_labels = test[j][1]
        outputs     = model(test_text).view(1, -1)
        predicted   = torch.argmax(outputs)
        # total    += 1
        correct    += (predicted == test_labels).sum()
      
      accuracy = 100.0 * correct / test_length
      print("Epoch: {}. Iters: {}. Loss: {}. Accuracy: {}.".format(epoch+1, k, loss.item(), accuracy))

    if last_loop: print("Epoch: {}. Iters: {}. Loss: {}. Accuracy: {}.".format(epoch+1, k, loss.item(), accuracy))

    batch_start += batch_size
    to_end = train_length - batch_end
    if to_end >= batch_size:
      batch_end += batch_size
    else:
      batch_end += to_end
      last_loop  = True