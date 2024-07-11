
import torch
import data_setup, model_builder, engine, utils
from torchvision import transforms
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import ToTensor

config = utils.read_config()

LEARNING_RATE = config['learning_rate']
WEIGHT_DECAY = config['weight_decay']
MOMENTUM = config['momentum']
BATCH_SIZE = config['batch_size']
NUM_EPOCHS = config['num_epochs']

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model_builder.Cifar10ModelV0().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY,
                            momentum=MOMENTUM)

train_dataloader, test_dataloader = data_setup.create_dataloaders(train_data, test_data, BATCH_SIZE)

engine.train(model, NUM_EPOCHS, train_dataloader, test_dataloader, optimizer, loss_fn, device, utils.accuracy_fn)

utils.save_model(model, "models", "CIFAR10_V0.pth")
