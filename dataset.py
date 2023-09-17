'''
Predict for next timestep // Seq2seq predict into future 
'''

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset, InMemoryDataset

# Init settings
Individual_path = "../Japan_Data/INDIVIDUAL"

class Japan_Dataset_Loader(object):
    def __init__(self, raw_data_dir=Individual_path) -> None:
        super(Japan_Dataset_Loader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self.process()
        
    def process(self):
        Ind_list = [folder for folder in os.listdir(self.raw_data_dir) if folder.endswith(".csv")]
        x_list, y_list = [], []

        for id in tqdm(Ind_list):
            tmp = pd.read_csv(os.path.join(Individual_path, id))
            if 999 not in tmp.iloc[:,-1].tolist():
                x_list.append(tmp.iloc[:,[6,7,8,9,10,13]])
                y_list.append(tmp.iloc[:,-1].tolist())

  
        x_stack = np.stack((x_list), axis=2)
        y_stack = np.array(y_list)
        x_stack = np.transpose(x_stack, (2, 0, 1))
        np.save('data/x_0711.npy', x_stack)
        np.save('data/y_0711.npy', y_stack)


JP_loader = Japan_Dataset_Loader()

