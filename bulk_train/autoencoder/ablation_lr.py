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
    head_dir = Path("weights/autoencoder/lr_adamw")

    # from 1 to 10 layers
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        train_autoencoder_with_hyperparameters(
            head_dir/f"lr{lr}", 
            tensorboard=SummaryWriter(f"runs/autoencoder/lr_adamw/lr{lr}"),
            channels_sizes=[(1,16), (16,8)], # medium size
            kernel_sizes=[7,7],
            dropout_rate=0.0,
            optimizer="adamw",
            learning_rate=lr,
            ) # all same kernel size
