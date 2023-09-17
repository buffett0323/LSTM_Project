import random
import torch 
import torch.nn as nn
from params import device

class LSTM_Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        '''We need to detach as we are doing truncated backpropagation through time (BPTT)
        If we don't, we'll backprop all the way to the start even after going through another batch'''
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(x.device)
        
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalization
        out = self.fc(out[:, -1, :])
        return out


class LSTM_Cluster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_prob=0.2):
        super(LSTM_Cluster, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = drop_prob).to(device)
        # self.bn = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        # out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalization
        out = self.fc(self.relu(out[:, -1, :]))
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_prob=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout = drop_prob).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(self.relu(out[:, -1, :]))  # Get the output from the last time step
        return out


class LSTM_MC(nn.Module):
    '''Ref code from: https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca'''
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_MC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, future=0, y=None):
        outputs = []
        # Reset the state of LSTM, the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            # Transform from 3d tensor to 2d tensor
            input_t = input_t.view(input_t.shape[0], input_t.shape[2]) 
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
             
        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs