'''
Predict for next timestep // Seq2seq predict into future 
'''

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from model import LSTM_Model, LSTM_MC
from itertools import chain
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


# Init settings
LOOKBACK = 5
batch_size = 32

# Sequence data
seq_data = np.load('data/seq_0717.npy', allow_pickle=True) # (6, 770400, 8)
seq_x = seq_data[:LOOKBACK, :, [1, 2, 3, 4, 5, 6, 7]].astype(np.int64) # Temporarily remove 0 (the time)
seq_y = seq_data[LOOKBACK, :, [6, 7]].astype(np.float64) # Temporarily predict the 6th 
seq_x = np.transpose(seq_x, (1, 0, 2))
seq_y = np.transpose(seq_y, (1, 0))
seq_y1 = seq_y[:, 0].reshape(-1, 1)
seq_y2 = seq_y[:, 1].reshape(-1, 1)


# Create a MinMaxScaler instance
scaler = MinMaxScaler()
seq_y1_scale = scaler.fit_transform(seq_y1) #.reshape(seq_y.shape[0], seq_y.shape[1])
seq_y2_scale = scaler.fit_transform(seq_y2) #.reshape(seq_y.shape[0], seq_y.shape[1])
merged_y = np.array([seq_y1_scale, seq_y2_scale]).reshape(seq_y1_scale.shape[0], 2)
max_y1, min_y1 = max(seq_y1), min(seq_y1)
max_y2, min_y2 = max(seq_y2), min(seq_y2)

X_train, X_sub, y_train, y_sub = train_test_split(seq_x, merged_y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)    

train_data = torch.Tensor(X_train)
train_labels = torch.Tensor(y_train).float()
val_data = torch.Tensor(X_val)
val_labels = torch.Tensor(y_val).float()
test_data = torch.Tensor(X_test)
test_labels = torch.Tensor(y_test).float()


# Model Pre-definition
model = LSTM_Model(input_size=seq_x.shape[2], hidden_size=64, num_layers=1, output_size=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


# Training loop
num_epochs = 50
best_val, patience = 1e10, 0
train_loss = []
for epoch in tqdm(range(num_epochs)):
    model.train()
    
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_labels)
    train_loss.append(loss.item())
    loss.backward()
    optimizer.step()
    
    scheduler.step()
    
    # Validation
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            outputs = model(val_data)
            # outputs = outputs.permute(0, 2, 1).contiguous()
            val_loss = criterion(outputs, val_labels)
            if val_loss.item() < best_val:
                best_val = val_loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break

# Plot the training loss
plt.plot(train_loss)
plt.title("Training Loss")
plt.show()

def region2id(y, thres=0.25):
    rec_i = 0
    for i in range(int(1/thres)):
        if y <= thres*i:
            break
        else:
            rec_i = i+1
    return rec_i


# Testing the model result
model.eval()
with torch.no_grad():
    outputs = model(test_data)
    test_loss = criterion(outputs, test_labels)
    print(test_loss.item())
    print(outputs)
    

    # Back to region
    test_region, pred_region = [], []
    for _, val in enumerate(test_labels):
        tmpx, tmpy = region2id(val[0]), region2id(val[1])
        test_region.append( (tmpx-1) * 5 + tmpy)
    
    for _, val in enumerate(outputs):
        tmpx, tmpy = region2id(val[0]), region2id(val[1])
        pred_region.append( (tmpx-1) * 5 + tmpy)
    
    
    accuracy = accuracy_score(test_region, pred_region)
    print(f'The accuracy is {round(accuracy*100, 2)} %')

    cm = confusion_matrix(test_region, pred_region)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp.plot()
    plt.show()


