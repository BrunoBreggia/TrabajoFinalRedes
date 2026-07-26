from bulk_train.train_func import train_autoencoder_with_hyperparameters
from pathlib import Path

if __name__=="__main__":
    head_dir= Path("weights/autoencoder/channels")

    train_autoencoder_with_hyperparameters(
        head_dir/"constant", 
        channels_sizes=[(1,1),(1,1),(1,1)])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"linear_increment", 
        channels_sizes=[(1,2),(2,3),(3,4)])
    
    train_autoencoder_with_hyperparameters(
        head_dir/"exp_increment", 
        channels_sizes=[(1,2),(2,4),(4,8)])