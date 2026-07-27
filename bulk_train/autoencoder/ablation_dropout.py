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
    head_dir= Path("weights/autoencoder/dropout")

    train_autoencoder_with_hyperparameters(
        head_dir/"prob0", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout/prob0"),
        dropout_rate=0.0)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob01", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout/prob01"),
        dropout_rate=0.1)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob02", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout/prob02"),
        dropout_rate=0.2)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob03", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout/prob03"),
        dropout_rate=0.3)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob04", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout/prob04"),
        dropout_rate=0.4)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob05", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout/prob05"),
        dropout_rate=0.5)