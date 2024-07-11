
import torch
from pathlib import Path
import configparser

def read_config():
    config = configparser.ConfigParser()
 
    config.read('config.ini')

    config_values = {
        'learning_rate': config.getfloat('PARAMETERS', 'learning_rate'),
        'weight_decay': config.getfloat('PARAMETERS', 'weight_decay'),
        'momentum': config.getfloat('PARAMETERS', 'momentum'),
        'batch_size': config.getint('PARAMETERS', 'batch_size'),
        'num_epochs': config.getint('PARAMETERS', 'num_epochs')}

    return config_values

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def save_model(model: torch.nn.Module,
               dir: str,
               model_name: str):

  MODEL_PATH = Path(dir)
  MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

  MODEL_SAVE_PATH = MODEL_PATH/model_name

  torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)
