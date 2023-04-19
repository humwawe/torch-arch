import torch
from torch.utils.data import DataLoader
from config import HP
import numpy as np


class DataSet(torch.utils.data.Dataset):
  def __init__(self, data_path):
    self.data_set = np.loadtxt(data_path, delimiter=',')

  def __getitem__(self, idx):
    item = self.data_set[idx]
    x, y = item[:HP.in_features], item[HP.in_features:]
    return torch.Tensor(x).float().to(HP.device), torch.Tensor(y).squeeze().long().to(HP.device)

  def __len__(self):
    return self.data_set.shape[0]
