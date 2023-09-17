'''
(Not use) Multi process
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
from model import LSTM_Model, LSTM_MC
from multiprocessing import Pool
from itertools import chain
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# Train test valid split
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(np.array(self.data[idx])), torch.FloatTensor(np.array([self.labels[idx]]))

    


def worker(rank, data_loader, model, criterion, optimizer):
    # Set the seed for reproducibility if needed
    torch.manual_seed(rank)
    
    train_loss = 0.0
    for batch in data_loader:
        train_loss += train_batch(model, criterion, optimizer, batch)
    return train_loss
    
def train_batch(model, criterion, optimizer, batch):
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, targets.reshape(targets.shape[0], targets.shape[2]))
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

        loss = criterion(outputs, targets.reshape(targets.shape[0], targets.shape[2]))
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
        print(outputs)
        
        # Back to region
        test_region, pred_region = [], []
        for _, val in enumerate(test_labels):
            tmpx, tmpy = region2id(val[0]), region2id(val[1])
            test_region.append( (tmpx-1) * 4 + tmpy)
        
        for _, val in enumerate(outputs):
            tmpx, tmpy = region2id(val[0]), region2id(val[1])
            pred_region.append( (tmpx-1) * 4 + tmpy)
        
        
        accuracy = accuracy_score(test_region, pred_region)
        print(f'The accuracy is {round(accuracy*100, 2)} %')

        cm = confusion_matrix(test_region, pred_region)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.show()


def region2id(y, thres=0.25):
    rec_i = 0
    for i in range(int(1/thres)):
        if y <= thres*i:
            break
        else:
            rec_i = i+1
    return rec_i





# Main
if __name__ == '__main__':
    # multiprocessing.freeze_support()
    
    # Hyperparameters
    LOOKBACK = 5
    batch_size = 64
    num_workers = 1
    num_epochs = 100
    best_val, patience = 1e10, 0
    
    # Sequence data
    seq_data = np.load('data/seq_0717.npy', allow_pickle=True) # (6, 770400, 8)
    seq_x = seq_data[:LOOKBACK, :, [1, 2, 3, 4, 5, 6, 7]].astype(np.int64) # Temporarily remove 0 (the time)
    seq_y = seq_data[LOOKBACK, :, [6, 7]].astype(np.float64) # Temporarily predict the 6th 
    seq_x = np.transpose(seq_x, (1, 0, 2))
    seq_y = np.transpose(seq_y, (1, 0))
    seq_y1 = seq_y[:, 0].reshape(-1, 1)
    seq_y2 = seq_y[:, 1].reshape(-1, 1)

    # Create a StandardScaler
    x_scaler = StandardScaler()
    seq_x_2d = seq_x.reshape((seq_x.shape[0] * seq_x.shape[1], seq_x.shape[2]))
    scaled_data_2d = x_scaler.fit_transform(seq_x_2d)
    seq_x_scaled = scaled_data_2d.reshape((seq_x.shape[0], seq_x.shape[1], seq_x.shape[2]))

    
    # Create a MinMaxScaler to y data
    scaler = MinMaxScaler()
    seq_y1_scale = scaler.fit_transform(seq_y1) #.reshape(seq_y.shape[0], seq_y.shape[1])
    seq_y2_scale = scaler.fit_transform(seq_y2) #.reshape(seq_y.shape[0], seq_y.shape[1])
    merged_y = np.array([seq_y1_scale, seq_y2_scale]).reshape(seq_y1_scale.shape[0], 2)
    max_y1, min_y1 = max(seq_y1), min(seq_y1)
    max_y2, min_y2 = max(seq_y2), min(seq_y2)
    

    # Train Test Valid split
    X_train, X_sub, y_train, y_sub = train_test_split(seq_x_scaled, merged_y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)    

    train_data = X_train
    train_labels = y_train
    val_data = torch.Tensor(X_val)
    val_labels = torch.Tensor(y_val).float()
    test_data = torch.Tensor(X_test)
    test_labels = torch.Tensor(y_test).float()
    
    
    '''
    test_region = []
    for _, val in enumerate(test_labels):
        tmpx, tmpy = region2id(val[0]), region2id(val[1])
        test_region.append( (tmpx-1) * 5 + tmpy)
    l = [(i, test_region.count(i)) for i in set(test_region)]
    print(l)
    '''
    
    # Create a DataLoader for your dataset
    dataset = CustomDataset(train_data, train_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create LSTM model, optimizer, and loss function
    model = LSTM_Model(input_size=seq_x.shape[2], hidden_size=64, num_layers=1, output_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_workers)
    train_losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()

        # res = pool.starmap(worker,  
        #   [(i, data_loader, model, criterion, optimizer) for i in range(num_workers)])
        
        tr_loss = training(data_loader, model, criterion, optimizer)
        print(tr_loss)
        train_losses.append(tr_loss)
        
        scheduler.step()
        
        # Early stopping Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(val_data)
                val_loss = criterion(outputs, val_labels)
                if val_loss.item() < best_val:
                    best_val = val_loss.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= 4:
                        break

    pool.close()
    pool.join()

    print("Training complete!")
    
    testing(model, criterion)


