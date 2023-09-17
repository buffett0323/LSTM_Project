import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
from dir_gnn_model import GNN
from dir_gnn_dataset import Japan_Dataset, Load_Dataset
from datetime import datetime
from tqdm import tqdm


def custom_collate(batch):
    return Batch.from_data_list(batch)
# Params
LOOKBACK = 5
CUT = 5

def training(data_loader, model, criterion, optimizer, l2_reg=False):
    model.train()
    train_loss = 0.0
    for batch in data_loader:
        targets = batch.y
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index)

        print(outputs.shape, targets.shape) 
        # torch.Size([640, 25]) torch.Size([128])
        loss = criterion(outputs, targets)#.flatten())
        
        # Apply L2 regularization through weight decay
        if l2_reg:
            l2_lambda = 0.001  # Adjust the L2 regularization strength
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss


def testing(data_loader, model, criterion):
    '''Not DONE YET'''
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs)
            test_loss += criterion(outputs, targets.flatten())    
            pred = torch.argmax(outputs, dim=1)
        
            # accuracy = accuracy_score(test_labels, pred)
            # print(f'The accuracy is {round(accuracy*100, 2)} %')

            # cm = confusion_matrix(test_labels, pred)
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
            # disp.plot()
            # plt.show()



if __name__ == '__main__':
    
    # Hyperparameters
    best_val, patience = 1e10, 0
    hidden_layers = 32
    num_layers = 2
    lr_rate = 0.1
    batch_size = 128
    num_epochs = 5
    
    # Train Test Split
    dataset = Load_Dataset(root='data/')

    # train_dataset, test_valid = train_test_split(dataset, test_size=0.2, random_state=42)
    # test_dataset, valid_dataset = train_test_split(test_valid, test_size=0.5, random_state=42)
    
    train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1
    train_size = int(train_ratio * len(dataset))
    test_size, valid_size = int((len(dataset) - train_size) / 2), int((len(dataset) - train_size) / 2)
    train_dataset, test_dataset, valid_dataset = random_split(dataset, [train_size, test_size, valid_size])


    # Create DataLoader instances for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=True)

    # Create GNN model, optimizer, and loss function
    model = GNN(num_features=6, num_classes=CUT*CUT, hidden_dim=hidden_layers, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss() # nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    

    # # Create a pool of worker processes
    # train_losses = []
    # for epoch in tqdm(range(num_epochs)):
    #     model.train()

    #     tr_loss = training(train_loader, model, criterion, optimizer)
    #     train_losses.append(tr_loss)
        
    #     scheduler.step()
        
        # # Early stopping Validation
        # if (epoch) % 5 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         outputs = model(val_data)
        #         val_loss = criterion(outputs, val_labels)
        #         if val_loss.item() < best_val:
        #             best_val = val_loss.item()
        #             patience = 0
        #         else:
        #             patience += 1
        #             if patience >= 5: break
    

'''

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
    num_epochs = 200
    
    
    # Sequence data
    seq_data = np.load('../data/seq_0717.npy', allow_pickle=True) # (6, 770400, 8)
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
    # model = LSTM_Cluster(input_size=seq_x.shape[2], hidden_size=hidden_layers, num_layers=num_layers, output_size=16)
    model = GRUModel(input_size=seq_x.shape[2], hidden_size=hidden_layers, num_layers=num_layers, output_size=16)
    criterion = nn.CrossEntropyLoss() # nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


    # Create a pool of worker processes
    train_losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()

        tr_loss = training(data_loader, model, criterion, optimizer)
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


'''