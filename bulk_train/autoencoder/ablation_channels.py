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
    head_dir= Path("weights/autoencoder/channels")

    train_autoencoder_with_hyperparameters(
        head_dir/"constant", 
        tensorboard=SummaryWriter(f"runs/autoencoder/channels/constant"),
        channels_sizes=[(1,1),(1,1),(1,1)])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"linear_increment", 
        tensorboard=SummaryWriter(f"runs/autoencoder/channels/linear_increment"),
        channels_sizes=[(1,2),(2,3),(3,4)])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"exp_increment", 
        tensorboard=SummaryWriter(f"runs/autoencoder/channels/exp_increment"),
        channels_sizes=[(1,2),(2,4),(4,8)])