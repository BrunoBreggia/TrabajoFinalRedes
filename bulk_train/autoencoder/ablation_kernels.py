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
    head_dir = Path("weights/autoencoder/kernels")

    train_autoencoder_with_hyperparameters(
        head_dir/"least", 
        tensorboard=SummaryWriter(f"runs/autoencoder/kernels/least"),
        kernel_sizes=[3,3,3])

    train_autoencoder_with_hyperparameters(
        head_dir/"most", 
        tensorboard=SummaryWriter(f"runs/autoencoder/kernels/most"),
        kernel_sizes=[7,7,7])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"incremental", 
        tensorboard=SummaryWriter(f"runs/autoencoder/kernels/incremental"),
        kernel_sizes=[3,5,7])

    train_autoencoder_with_hyperparameters(
        head_dir/"deremental", 
        tensorboard=SummaryWriter(f"runs/autoencoder/kernels/decremental"),
        kernel_sizes=[7,5,3])