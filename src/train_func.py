import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import sys
# 1. Calculate the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)
# 2. Append it to Python's search paths if it isn't already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.modulos import Autoencoder, Encoder, ModuloClasificador, ClasificadorCompleto
from src.dataset import fashion_dataset
from src.utils import TrainerEncoder, TrainerClassifier, save_model

def train_and_test_classifier_with_hyperparameters(
        dir_path:str,
        path_encoder:str,
        freeze_encoder:bool=True,
        tensorboard=None,
        channels_sizes:tuple=((8,30),(30,20),(20,10)), # variable
        kernel_sizes:tuple=(7,7,7), # variable
        dropout_rate:float=0.0, # variable
    ):
    # Create directory if doesnt exist
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Assemble encoder (with best configuration from ablation)
    encoder = Encoder(layers=2, 
                      channels_sizes=[(1,16), (16,8)],
                      kernel_sizes=[7,7], 
                      dropoout_rate=0.0,
                      pool=True)
    state_dict = torch.load(path_encoder, weights_only=True)
    encoder.state_dict(state_dict)

    # Assemble classifier
    clasificador = ModuloClasificador(channels_sizes=channels_sizes,  
                                      kernel_sizes=kernel_sizes,
                                      dropout_rate=dropout_rate)

    # Build model (autoencoder + classifier)
    model = ClasificadorCompleto(encoder, clasificador, freeze_encoder)

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
    learning_rate = 0.001
    optim = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-2)
    loss_func = nn.CrossEntropyLoss()
    trainer = TrainerClassifier(model=model,
                            tensorboard=tensorboard,
                            train_dataloader=train_loader,
                            optimizer=optim,
                            loss_criterion=loss_func,
                            val_dataloader=val_loader,
                            early_stopping_patience=None)

    # Train
    trainer.fit(max_epochs=20, verbose=True)
    # Evaluate and save confusion matrix
    trainer.test(test_loader, verbose=True, confusionMat_dir=dir_path)
    #save_model(model, dir_path)

def train_autoencoder_with_hyperparameters(
        dir_path:str,
        tensorboard=None,
        layers:int=2,
        channels_sizes:tuple=((1,32),(32,8)), # variable
        kernel_sizes:tuple=(3,7),
        dropout_rate:float=0.0, # variable
        learning_rate:float=1e-1, # variable
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
    trainer = TrainerEncoder(model=model,
                    tensorboard=tensorboard,
                    train_dataloader=train_loader,
                    optimizer=optim,
                    loss_criterion=loss_func,
                    val_dataloader=val_loader,
                    early_stopping_patience=None)

    trainer.fit(max_epochs=20, verbose=True)
    save_model(model.encoder, dir_path)

if __name__=="__main__":
    writer = SummaryWriter("runs/trial02")
    #train_autoencoder_with_hyperparameters(dir_path="weights/exp1", tensorboard=writer)
    train_and_test_classifier_with_hyperparameters(
        dir_path="weights/exp2",
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        freeze_encoder=True,
        tensorboard=None,
    )
