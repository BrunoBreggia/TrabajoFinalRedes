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
    head_dir= Path("weights/classifier/channel")

    train_and_test_classifier_with_hyperparameters(
        head_dir/"large", 
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        tensorboard=SummaryWriter(f"runs/classifier/channels/large"),
        channels_sizes=[(8,30),(30,30),(30,10)])
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"medium", 
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        tensorboard=SummaryWriter(f"runs/classifier/channels/medium"),
        channels_sizes=[(8,30),(30,20),(20,10)])
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"constant_end", 
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        tensorboard=SummaryWriter(f"runs/classifier/channels/constant_end"),
        channels_sizes=[(8,10),(10,10),(10,10)])
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"constant_initial", 
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        tensorboard=SummaryWriter(f"runs/classifier/channels/constant_initial"),
        channels_sizes=[(8,7),(7,7),(7,10)])