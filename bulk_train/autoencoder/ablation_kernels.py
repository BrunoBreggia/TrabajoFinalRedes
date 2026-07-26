from bulk_train.train_func import train_autoencoder_with_hyperparameters
from pathlib import Path

if __name__=="__main__":
    head_dir = Path("weights/autoencoder/kernels")

    train_autoencoder_with_hyperparameters(
        head_dir/"least", 
        kernel_sizes=[3,3,3])

    train_autoencoder_with_hyperparameters(
        head_dir/"most", 
        kernel_sizes=[7,7,7])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"incremental", 
        kernel_sizes=[3,5,7])

    train_autoencoder_with_hyperparameters(
        head_dir/"deremental", 
        kernel_sizes=[7,5,3])