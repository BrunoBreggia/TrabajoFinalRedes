from bulk_train.train_func import train_autoencoder_with_hyperparameters
from pathlib import Path

if __name__=="__main__":
    head_dir = Path("weights/autoencoder/layers")

    # from 1 to 10 layers
    for i in range(1,11):
        train_autoencoder_with_hyperparameters(
            head_dir/f"layers{i}", 
            layers=i,
            channels_sizes=[(2**k,2**(k+1)) for k in range(0,i)], # exponential increment
            kernel_sizes=[3]*i) # all same kernel size