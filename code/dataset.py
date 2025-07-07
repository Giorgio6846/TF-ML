import lightning as L
from torch.utils.data import Dataset, DataLoader

class PriceDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class LitPriceData(L.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=64):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = PriceDataset(self.X_train, self.y_train)
        self.val_dataset = PriceDataset(self.X_val, self.y_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)