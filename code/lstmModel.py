import torch.nn as nn
import lightning as L

class LSTMModel(L.LightningModule):
    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 7)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x