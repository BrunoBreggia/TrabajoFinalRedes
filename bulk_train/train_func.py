import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split

from pathlib import Path
import sys
# 1. Calculate the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)
# 2. Append it to Python's search paths if it isn't already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.modulos import Autoencoder
from src.dataset import fashion_dataset
from src.utils import Trainer, save_model

def train_autoencoder_with_hyperparameters(
        dir_path:str,
        layers:int=3,
        channels_sizes:tuple=((1,2),(2,4),(4,8)),
        kernel_sizes:tuple=(3,3,3),
        dropout_rate:float=0.0,
        learning_rate:float=1e-3,
        optimizer:str="sgd",
    ):
    # Create directory if doesnt exist
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    # Build model (autoencoder)
    model = Autoencoder(layers=layers, 
                        channels_sizes=channels_sizes, 
                        kernel_sizes=kernel_sizes,
                        dropout_rate=dropout_rate)
    # Download train data
    train_data = FashionMNIST(root="data/mnist-data", 
                              train=True,
                              transform=transforms.ToTensor(), 
                              download=True)
    # 80/20 partition for train & validacion data
    val_size = int(len(train_data)*.2)
    train_data, val_data = random_split(train_data, [len(train_data) - val_size, val_size])
    # Donwload test data
    test_data = FashionMNIST(root="data/mnist-data", 
                             train=False,
                             transform=transforms.ToTensor(), 
                             download=True)
    # Build dataset objects
    train_data = fashion_dataset(train_data)
    val_data = fashion_dataset(val_data)
    test_data  = fashion_dataset(test_data)
    # Build dataloaders
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    # Configure trainer
    if optimizer.lower() == "sgd":
        optim = torch.optim.SGD(params=model.parameters(),lr=learning_rate)
    elif optimizer.lower() == "sgd nesterov":
        optim = torch.optim.SGD(params=model.parameters(),lr=learning_rate,momentum=0.9,nesterov=True,weight_decay=1e-4)
    elif optimizer.lower() == "adam":
        optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    elif optimizer.lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-2)
    loss_func = nn.MSELoss()
    trainer = Trainer(model=model,
                    model_type="autoencoder",
                    train_dataloader=train_loader,
                    optimizer=optim,
                    loss_criterion=loss_func,
                    val_dataloader=val_loader,
                    early_stopping_patience=None)

    trainer.fit(max_epochs=5, verbose=True)
    save_model(model, dir_path)

if __name__=="__main__":
    train_autoencoder_with_hyperparameters(dir_path="weights/exp1")
