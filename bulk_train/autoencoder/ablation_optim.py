from bulk_train.train_func import train_autoencoder_with_hyperparameters
from pathlib import Path

if __name__=="__main__":
    head_dir= Path("weights/autoencoder/ablation_optim")

    train_autoencoder_with_hyperparameters(
        head_dir/"sgd", 
        optimizer="sgd")

    train_autoencoder_with_hyperparameters(
        head_dir/"sgd_nesterov", 
        optimizer="sgd nesterov")

    train_autoencoder_with_hyperparameters(
        head_dir/"adam", 
        optimizer="adam")

    train_autoencoder_with_hyperparameters(
        head_dir/"adamw", 
        optimizer="adamw")
