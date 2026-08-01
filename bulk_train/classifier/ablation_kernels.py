from pathlib import Path
import sys
# 1. Calculate the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
# 2. Append it to Python's search paths if it isn't already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.train_func import train_and_test_classifier_with_hyperparameters
from torch.utils.tensorboard import SummaryWriter

if __name__=="__main__":
    head_dir = Path("weights/classifier/kernels")

    train_and_test_classifier_with_hyperparameters(
        head_dir/"least", 
        tensorboard=SummaryWriter(f"runs/classifier/kernels/least"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[3,3,3])

    train_and_test_classifier_with_hyperparameters(
        head_dir/"most", 
        tensorboard=SummaryWriter(f"runs/classifier/kernels/most"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[7,7,7])
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"incremental", 
        tensorboard=SummaryWriter(f"runs/classifier/kernels/incremental"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[3,5,7])

    train_and_test_classifier_with_hyperparameters(
        head_dir/"deremental", 
        tensorboard=SummaryWriter(f"runs/classifier/kernels/decremental"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[7,5,3])