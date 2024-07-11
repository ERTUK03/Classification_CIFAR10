
from torch.utils.data import DataLoader
import torchvision

def create_dataloaders(
    train_data: torchvision.datasets,
    test_data: torchvision.datasets,
    batch_size: int):

  train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True)

  test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False)

  return train_dataloader, test_dataloader
