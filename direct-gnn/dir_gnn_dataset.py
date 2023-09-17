import torch
import pandas as pd
import numpy as np
import math
import os
import os.path as osp
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



class Japan_Dataset(Dataset):
    def __init__(self, root, lookback, cut, process_need=False):
        self.data = np.transpose(np.load('data/seq_6.npy', allow_pickle=True), (1, 0, 2))
        self.seq_length = lookback
        self.root = root
        self.cut = cut
        self.x_shape = 7
        self.x_data = self.data[:, :self.seq_length, [1, 2, 3, 4, 5, 6, 7]].astype(np.float64)
        
        if process_need:
            self.adj_info = self._get_adjacency_info()
            self.y_cluster = self.y_pre_process()
            self.x_pre_process()    
            self.process()

        super(Japan_Dataset, self).__init__(root, lookback, cut, process_need)

    @property
    def raw_file_names(self):
        return 'seq_6.npy'

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.data.shape[0])]
    
    
    def download(self):
        pass

    def x_pre_process(self):
        # Create a StandardScaler
        x_scaler = StandardScaler()
        seq_x_2d = self.x_data.reshape((self.x_data.shape[0] * self.x_data.shape[1], self.x_data.shape[2]))
        scaled_data_2d = x_scaler.fit_transform(seq_x_2d)
        self.x_data = scaled_data_2d.reshape((self.x_data.shape[0], self.x_data.shape[1], self.x_data.shape[2]))

        
    def y_pre_process(self):
        # Create a MinMaxScaler to y data
        scaler = MinMaxScaler()
        seq_y = self.data[:, self.seq_length, [6, 7]].astype(np.float64) # Temporarily predict the 6th 
        seq_y = np.transpose(seq_y, (0, 1))
        
        seq_y1 = seq_y[:, 0].reshape(-1, 1)
        seq_y2 = seq_y[:, 1].reshape(-1, 1)
        seq_y1_scale = scaler.fit_transform(seq_y1) #.reshape(seq_y.shape[0], seq_y.shape[1])
        seq_y2_scale = scaler.fit_transform(seq_y2) #.reshape(seq_y.shape[0], seq_y.shape[1])
        return [self.region2id(y1) * self.cut + self.region2id(y2) for y1, y2 in zip(seq_y1_scale, seq_y2_scale)]
    
        
    def process(self):
        for index, matrix in tqdm(enumerate(self.data), total=self.data.shape[0]):
            matrix = pd.DataFrame(matrix)
            edge_feats = self._get_edge_features(matrix)
            node_feats = self._get_node_features(self.x_data[index])
            label = self._get_labels(self.y_cluster[index])
            
            # Create data object
            data = Data(x=node_feats, 
                        edge_index=self.adj_info,
                        edge_attr=edge_feats,
                        y=label)

            # Store data
            # if not os.path.exists(os.path.join(self.store_path, f'data_{index}.pt')):
            torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))


    def _get_node_features(self, mat):
        all_node_feats = []
        mat = pd.DataFrame(mat)
        for i in range(self.seq_length):
            all_node_feats.append(mat.iloc[i, 1:].tolist())
        
        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)


    def _get_labels(self, label):
        return torch.tensor(label, dtype=torch.int64)
    

    def _get_adjacency_info(self):
        '''This will return a directed graph'''
        edge_indices = [[i, i+1] for i in range(self.seq_length - 1)]
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices
    
    
    def _get_edge_features(self, mat):
        '''This will return a matrix including time jetlag, distance'''
        # Datetime process
        datetime_row = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in mat.iloc[0:self.seq_length, 0]]
        time_list = [int((datetime_row[i+1] - datetime_row[i]).seconds / 60 )for i in range(self.seq_length-1)]

        # Distance
        position = [(lng, lat) for lng, lat in zip(mat.iloc[0:self.seq_length, -2], mat.iloc[0:self.seq_length, -1])]
        dist_list = [self._get_distance(position[i+1], position[i]) for i in range(self.seq_length-1)]
        
        vec1 = np.array(time_list).reshape(1, -1)
        vec2 = np.array(dist_list).reshape(1, -1)
        all_edge_feats = np.concatenate((vec1, vec2), axis=0)
        return torch.tensor(all_edge_feats, dtype=torch.float)
        
            
    def _get_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x2-x1) ** 2 + (y2-y1) ** 2) ** 0.5
    
    
    def region2id(self, y):
        thres = float(1/self.cut)
        thres_range = [float(i * thres) for i in range(self.cut)]
        id = 0
        for i, thres in enumerate(thres_range):
            if y >= thres: id = i
        return id   

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

# dataset = Japan_Dataset(root='data/', lookback=5, cut=5, process_need=False)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for batch in data_loader:
#     print(batch)

# Process data
# Japan_Dataset(root='data/', lookback=5, cut=5, process_need=True)


class Load_Dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.data = np.transpose(np.load('data/seq_6.npy', allow_pickle=True), (1, 0, 2))
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return 'seq_6.npy'

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.data.shape[0])]
    
    def download(self):
        pass

    def process(self):
        pass
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


