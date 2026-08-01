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
    head_dir= Path("weights/autoencoder/channel")

    train_autoencoder_with_hyperparameters(
        head_dir/"large", 
        tensorboard=SummaryWriter(f"runs/autoencoder/channels/large"),
        channels_sizes=[(1,32),(32,8)])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"medium", 
        tensorboard=SummaryWriter(f"runs/autoencoder/channels/medium"),
        channels_sizes=[(1,16),(16,8)])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"small", 
        tensorboard=SummaryWriter(f"runs/autoencoder/channels/small"),
        channels_sizes=[(1,8),(8,8)])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"bottleneck", 
        tensorboard=SummaryWriter(f"runs/autoencoder/channels/bottleneck"),
        channels_sizes=[(1,4),(4,8)])