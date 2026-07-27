import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.modulos import Autoencoder
from src.dataset import fashion_dataset
from src.utils import TrainerEncoder, save_model

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/trial01")

# Build model (autoencoder)
model = Autoencoder(layers=3, 
                    channels_sizes=[(1,8), (8,16), (16,32)], 
                    kernel_sizes=[3,5,7],
                    dropout_rate=0.2)

# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Model parameters: {trainable_params}")

# Build datasets
train_data = FashionMNIST(root="data/mnist-data", 
                          train=True,
                          transform=transforms.ToTensor(), 
                          download=True)

# particionado 80/20 para train y validacion
val_size = int(len(train_data)*.2)
train_data, val_data = random_split(train_data, [len(train_data) - val_size, val_size])

test_data = FashionMNIST(root="data/mnist-data", 
                         train=False,
                         transform=transforms.ToTensor(), 
                         download=True)

train_data = fashion_dataset(train_data)
val_data = fashion_dataset(val_data)
test_data  = fashion_dataset(test_data)

# Build dataloaders
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# Configure trainer
lr = 0.01
optim = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.MSELoss()

trainer = TrainerEncoder(model=model,
                         tensorboard=writer,
                         train_dataloader=train_loader,
                         optimizer=optim,
                         loss_criterion=loss_func,
                         val_dataloader=val_loader,
                         early_stopping_patience=None)

trainer.fit(max_epochs=5, verbose=True)
save_model(model.encoder, Path("weights/trial_01"))
