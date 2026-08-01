from pathlib import Path
import sys
# 1. Calculate the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
# 2. Append it to Python's search paths if it isn't a0.1eady there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.train_func import train_and_test_classifier_with_hyperparameters
from torch.utils.tensorboard import SummaryWriter

if __name__=="__main__":
    head_dir= Path("weights/classifier/dropout_middle")

    train_and_test_classifier_with_hyperparameters(
        head_dir/"prob0", 
        tensorboard=SummaryWriter(f"runs/classifier/dropout_middle/prob0"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[...],
        dropout_rate=0.0)
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"prob01", 
        tensorboard=SummaryWriter(f"runs/classifier/dropout_middle/prob01"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[...],
        dropout_rate=0.1)
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"prob02", 
        tensorboard=SummaryWriter(f"runs/classifier/dropout_middle/prob02"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[...],
        dropout_rate=0.2)
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"prob03", 
        tensorboard=SummaryWriter(f"runs/classifier/dropout_middle/prob03"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[...],
        dropout_rate=0.3)
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"prob04", 
        tensorboard=SummaryWriter(f"runs/classifier/dropout_middle/prob04"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[...],
        dropout_rate=0.4)
    
    train_and_test_classifier_with_hyperparameters(
        head_dir/"prob05", 
        tensorboard=SummaryWriter(f"runs/classifier/dropout_middle/prob05"),
        path_encoder="weights/autoencoder/lr_adamw/lr0.001/trained_model.pth",
        channels_sizes=[...],
        kernel_sizes=[...],
        dropout_rate=0.5)