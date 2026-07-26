from bulk_train.train_func import train_autoencoder_with_hyperparameters
from pathlib import Path

if __name__=="__main__":
    head_dir= Path("weights/autoencoder/dropout")

    train_autoencoder_with_hyperparameters(
        head_dir/"prob0", 
        dropout_rate=0.0)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob01", 
        dropout_rate=0.1)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob02", 
        dropout_rate=0.2)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob03", 
        dropout_rate=0.3)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob04", 
        dropout_rate=0.4)
    
    train_autoencoder_with_hyperparameters(
        head_dir/"prob05", 
        dropout_rate=0.5)