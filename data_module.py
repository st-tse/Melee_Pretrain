import torch
from torch.utils.data import DataLoader, Dataset

class FrameDataset(Dataset):
    def __init__(self,X,y):
        self.x_train=torch.tensor(X,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)
        
    def __len__(self):
        return len(self.y_train)
  
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]