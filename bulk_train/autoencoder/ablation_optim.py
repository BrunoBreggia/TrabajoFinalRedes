from pathlib import Path
import sys
# 1. Calculate the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
# 2. Append it to Python's search paths if it isn't already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.train_func import train_autoencoder_with_hyperparameters
from torch.utils.tensorboard import SummaryWriter

if __name__=="__main__":
    head_dir= Path("weights/autoencoder/ablation_optim")

    train_autoencoder_with_hyperparameters(
        head_dir/"sgd", 
        tensorboard=SummaryWriter(f"runs/autoencoder/optim/sgd"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.0,
        optimizer="sgd")

    train_autoencoder_with_hyperparameters(
        head_dir/"sgd_nesterov", 
        tensorboard=SummaryWriter(f"runs/autoencoder/optim/sgd_nesterov"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.0,
        optimizer="sgd nesterov")

    train_autoencoder_with_hyperparameters(
        head_dir/"adam", 
        tensorboard=SummaryWriter(f"runs/autoencoder/optim/adam"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.0,
        optimizer="adam")

    train_autoencoder_with_hyperparameters(
        head_dir/"adamw", 
        tensorboard=SummaryWriter(f"runs/autoencoder/optim/adamw"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.0,
        optimizer="adamw")
