import lightning as L
import torch.nn.functional as F
import torch

from lstmModel import LSTMModel

class LitTrainer(L.LightningModule):
    def __init__(self, hidden_size, num_layers, dropout, input_size=1, lr=1e-3):
        super().__init__()
        self.model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.lr = lr
    
    def training_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = F.mse_loss(out, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer