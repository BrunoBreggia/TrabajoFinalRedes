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
    head_dir = Path("weights/autoencoder/layers_decrement")

    # from 1 to 10 layers
    for i in range(1,11):
        train_autoencoder_with_hyperparameters(
            head_dir/f"layers{i}", 
            tensorboard=SummaryWriter(f"runs/autoencoder/layers_decrement/layers{i}"),
            layers=i,
            channels_sizes=[(2**k,2**(k+1)) for k in range(0,i)], # exponential increment
            kernel_sizes=[3]*i) # all same kernel size
