'''
(Not use) Multi process clustering
'''
import os
import torch
import multiprocessing
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from model import LSTM_Model, LSTM_MC, LSTM_Cluster
from multiprocessing import Pool
from itertools import chain
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from datetime import datetime



# Train test valid split
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(np.array(self.data[idx])), torch.LongTensor(np.array([self.labels[idx]]))




# Worker
def worker(rank, data_loader, model, criterion, optimizer):
    torch.manual_seed(rank) # Set the seed for reproducibility if needed
    
    train_loss = 0.0
    for batch in data_loader:
        train_loss += train_batch(model, criterion, optimizer, batch)
    return train_loss



# Train for one batch
def train_batch(model, criterion, optimizer, batch):
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets.flatten())
    
    # Apply L2 regularization through weight decay
    l2_lambda = 0.001  
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += l2_lambda * l2_reg
    
    loss.backward()
    optimizer.step()
    return loss.item()




# Training without multiprocess
def training(data_loader, model, criterion, optimizer):
    train_loss = 0.0
    for batch in data_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets.flatten())
        
        # Apply L2 regularization through weight decay
        l2_lambda = 0.001  # Adjust the L2 regularization strength
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        
        loss += l2_lambda * l2_reg
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss
   
    
# Testing the model result
def testing(model, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        test_loss = criterion(outputs, test_labels)
        print(test_loss.item())
        
        pred = torch.argmax(outputs, dim=1)
        
        accuracy = accuracy_score(test_labels, pred)
        print(f'The accuracy is {round(accuracy*100, 2)} %')

        cm = confusion_matrix(test_labels, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.show()

# Y Transformation
def region2id(y, thres=0.25):
    thres_range = [float(i * thres) for i in range(int(1/thres))]
    id = 0
    for i, thres in enumerate(thres_range):
        if y >= thres:
            id = i

    return id





# Main
if __name__ == '__main__':
    
    # Parameters
    LOOKBACK = 5
    best_val, patience = 1e10, 0
    
    # Hyperparams
    hidden_layers = 64
    num_layers = 2
    lr_rate = 0.1
    batch_size = 64
    num_epochs = 10 # 200
    num_workers = 4 # 16
    
    
    # Sequence data
    seq_data = np.load('data/seq_0717.npy', allow_pickle=True) # (6, 770400, 8)
    seq_x = seq_data[:LOOKBACK, :, [1, 2, 3, 4, 5, 6, 7]].astype(np.int64) # Temporarily remove 0 (the time)
    seq_y = seq_data[LOOKBACK, :, [6, 7]].astype(np.float64) # Temporarily predict the 6th 
    seq_x = np.transpose(seq_x, (1, 0, 2))
    seq_y = np.transpose(seq_y, (1, 0))
    
    # Time lag
    added_x = np.transpose(seq_data[:, :, 0], (1, 0))
    datetime_array = []
    for row in added_x:
        datetime_row = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in row]
        diff_list = [int((datetime_row[i+1] - datetime_row[i]).seconds / 60 )for i in range(len(datetime_row)-1)]
        datetime_array.append(diff_list)
    
    # Concatenate the two arrays along the new axis (axis=2)
    dt_arr = np.expand_dims(np.array(datetime_array), axis=2)
    seq_x = np.concatenate((dt_arr, seq_x), axis=2)
    

    # Create a StandardScaler
    x_scaler = StandardScaler()
    seq_x_2d = seq_x.reshape((seq_x.shape[0] * seq_x.shape[1], seq_x.shape[2]))
    scaled_data_2d = x_scaler.fit_transform(seq_x_2d)
    seq_x_scaled = scaled_data_2d.reshape((seq_x.shape[0], seq_x.shape[1], seq_x.shape[2]))

    
    # Create a MinMaxScaler to y data
    scaler = MinMaxScaler()
    seq_y1 = seq_y[:, 0].reshape(-1, 1)
    seq_y2 = seq_y[:, 1].reshape(-1, 1)
    seq_y1_scale = scaler.fit_transform(seq_y1) #.reshape(seq_y.shape[0], seq_y.shape[1])
    seq_y2_scale = scaler.fit_transform(seq_y2) #.reshape(seq_y.shape[0], seq_y.shape[1])
    y_cluster = [region2id(y1) * 4 + region2id(y2) for y1, y2 in zip(seq_y1_scale, seq_y2_scale)]
    
    
    # Train Test Valid split
    X_train, X_sub, y_train, y_sub = train_test_split(seq_x_scaled, y_cluster, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)    

    train_data = X_train
    train_labels = y_train
    val_data = torch.Tensor(X_val)
    val_labels = torch.LongTensor(y_val)#.float()
    test_data = torch.Tensor(X_test)
    test_labels = torch.LongTensor(y_test)#.float()
    
    
    # Create a DataLoader for your dataset
    dataset = CustomDataset(train_data, train_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Create LSTM model, optimizer, and loss function
    model = LSTM_Cluster(input_size=seq_x.shape[2], hidden_size=hidden_layers, num_layers=num_layers, output_size=16)
    criterion = nn.CrossEntropyLoss() # nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_workers)
    train_losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()

        res = pool.starmap(
            worker,
            [(i, data_loader, model, criterion, optimizer) for i in range(num_workers)])
        
        tr_loss = sum(res)
        
        # tr_loss = training(data_loader, model, criterion, optimizer)
        train_losses.append(tr_loss)
        
        scheduler.step()
        
        # Early stopping Validation
        if (epoch) % 5 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(val_data)
                val_loss = criterion(outputs, val_labels)
                if val_loss.item() < best_val:
                    best_val = val_loss.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5: break
    
    
    # Plot the training loss
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.show()

    print("Training complete!")
    
    testing(model, criterion)


