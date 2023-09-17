import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from model import LSTM_Model, LSTM_MC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from imblearn.over_sampling import SMOTE

# Init settings
x = np.load('data/x_0711.npy') 
y = np.load('data/y_0711.npy') 
num_classes = 9

X_train, X_sub, y_train, y_sub = train_test_split(x, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)
print(x.shape, X_train.shape)
# Model 
model = LSTM_MC(input_size=x.shape[2], hidden_size=10, output_size=num_classes)
criterion = nn.CrossEntropyLoss() # nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) # torch.optim.Adam(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_data = torch.Tensor(X_train)
train_labels = torch.Tensor(y_train).long() 
val_data = torch.Tensor(X_val)
val_labels = torch.Tensor(y_val).long()
test_data = torch.Tensor(X_test)
test_labels = torch.Tensor(y_test).long()

# Training loop
num_epochs = 100
best_val, patience = 1e10, 0
train_loss = []
for epoch in tqdm(range(num_epochs)):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_data)
    outputs = outputs.permute(0, 2, 1).contiguous()
    
    loss = criterion(outputs, train_labels)
    train_loss.append(loss.item())

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Validation
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            outputs = model(val_data)
            outputs = outputs.permute(0, 2, 1).contiguous()
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
    
# Eval
model.eval()
with torch.no_grad():
    outputs = model(test_data)
    pred = torch.argmax(outputs, dim=2)
    y_test, y_pred = test_labels.tolist(), pred.tolist()
    y_test, y_pred = list(chain.from_iterable(y_test)), list(chain.from_iterable(y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f'The accuracy is {round(accuracy*100, 2)} %')

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=None)
    disp.plot()
    plt.show()

    
    

