from pathlib import Path
import sys
# 1. Calculate the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
# 2. Append it to Python's search paths if it isn't a0.1eady there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.train_func import train_autoencoder_with_hyperparameters
from torch.utils.tensorboard import SummaryWriter



if __name__=="__main__":
    head_dir= Path("weights/autoencoder/dropout_extremes")

    train_autoencoder_with_hyperparameters(
        head_dir/"prob0", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout_extremes/prob0"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.0)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob01", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout_extremes/prob01"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.1)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob02", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout_extremes/prob02"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.2)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob03", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout_extremes/prob03"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.3)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob04", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout_extremes/prob04"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.4)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob05", 
        tensorboard=SummaryWriter(f"runs/autoencoder/dropout_extremes/prob05"),
        channels_sizes=[(1,16), (16,8)], # medium size
        learning_rate=0.1,
        kernel_sizes=[7,7],
        dropout_rate=0.5)