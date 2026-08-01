import torch
from torch import nn

class ClasificadorCompleto(nn.Module):
    def __init__(self, encoder, clasificador, freeze_encoder=True):
        super().__init__()

        self.encoder = encoder
        self.clasificador = clasificador

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.clasificador(x)
        return x
        

class ModuloClasificador(nn.Module):
    def __init__(self, channels_sizes, kernel_sizes, dropout_rate=0.0, device=None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        assert len(channels_sizes) == len(kernel_sizes) == 3, "There are three layers in the Classifier, give data for these three and only these three layers"
        assert channels_sizes[0][0] == 8, "Fisrt channel must be dimension 8"
        assert channels_sizes[2][1] == 10, "Last channel must be dimension 10"
        
        # parametros
        self.device = device
        self.channels_sizes = channels_sizes
        self.kernel_sizes = kernel_sizes

        # first convolution
        self.conv1 = nn.Conv2d(in_channels=channels_sizes[0][0],
                                out_channels=channels_sizes[0][1],
                                kernel_size=kernel_sizes[0],
                                padding=(kernel_sizes[0]-3)//2,
                                bias=False,
                                )
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(channels_sizes[0][1]) # matches out_channels of the Conv layer

        # second convolution (w/dropout)
        self.conv2 = nn.Conv2d(in_channels=channels_sizes[1][0],
                                out_channels=channels_sizes[1][1],
                                kernel_size=kernel_sizes[1],
                                padding=(kernel_sizes[1]-3)//2,
                                bias=False,
                                )
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm2d(channels_sizes[1][1]) # matches out_channels of the Conv layer
        self.spatial_dropout2 = nn.Dropout2d(p=dropout_rate)  # low rate for CNN layers, like 0.2

        # third convolution
        self.conv3 = nn.Conv2d(in_channels=channels_sizes[2][0],
                                out_channels=channels_sizes[2][1],
                                kernel_size=kernel_sizes[2],
                                padding=(kernel_sizes[2]-3)//2,
                                bias=False,
                                )
        self.relu3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm2d(channels_sizes[2][1]) # matches out_channels of the Conv layer

    def forward(self, x):
        # first convolution
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)

        # second convolution
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)
        x = self.spatial_dropout2(x)

        # third convolution
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batch_norm3(x)

        # reshape
        x = x.reshape(-1,10)  # 1 x 10 tensor

        return x


class Autoencoder(nn.Module):
    def __init__(self, layers, channels_sizes, kernel_sizes, device=None, **kwargs):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            #print(f"Usando dispositivo: {device.upper()}")

        # parametros
        self.device = device
        self.layers = layers
        self.channels_sizes = channels_sizes
        self.kernel_sizes = kernel_sizes
        self.config = kwargs

        # instantiate encoder
        self.encoder = Encoder(layers, channels_sizes, kernel_sizes, **kwargs)
        # instantiate decoder
        self.decoder = Decoder(
            layers, 
            [pair[::-1] for pair in channels_sizes[::-1]], 
            kernel_sizes[::-1],
            **kwargs
            )
        
        self.to(device)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class Encoder(nn.Module):
    def __init__(self, layers, channels_sizes, kernel_sizes, **kwargs):
        super().__init__()

        assert layers == len(channels_sizes), "Amount of layers must equal pairs of channels dimensions"
        assert layers == len(kernel_sizes), "Amount of layers must equal amount of kernel sizes"

        self.layers = nn.Sequential(*[
            DownsampleBlock(
                in_channels=channels_sizes[i][0],
                out_channels=channels_sizes[i][1],
                kernel_size=kernel_sizes[i],
                **kwargs,
                ) for i in range(layers)
            ])

    def forward(self, x):
        x = self.layers(x)
        return x

class Decoder(nn.Module):
    def __init__(self, layers, channels_sizes, kernel_sizes, **kwargs):
        super().__init__()

        assert layers == len(channels_sizes), "Amount of layers must equal pairs of channels dimensions"
        assert layers == len(kernel_sizes), "Amount of layers must equal amount of kernel sizes"

        self.layers = nn.Sequential(*[
            UpsampleBlock(
                in_channels=channels_sizes[i][0],
                out_channels=channels_sizes[i][1],
                kernel_size=kernel_sizes[i],
                **kwargs,
                ) for i in range(layers)
            ])

    def forward(self, x):
        x = self.layers(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.2, **kwargs):
        super().__init__()

        # Error if kernel size is odd
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        # layers
        #self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        pad = (kernel_size-1)//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.spatial_dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        #print(f"Size before Upsampler:{x.shape}")
        # 1. Upsample (learnable)
        x = self.upsample(x)

        # 2. Learnable Feature Processing
        x = self.conv(x)

        # 3. Non-linearity
        x = self.relu(x)

        # 4. Normalization
        x = self.batch_norm(x)

        # 5. Regularization (Spatial Dropout)
        x = self.spatial_dropout(x)
        #print(f"Size after Upsampler:{x.shape}")
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.2, pool=True, **kwargs):
        super().__init__()

        # Error if kernel size is odd
        assert kernel_size > 1, "Kernel size must be larger than 1"
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        # layers
        pad = (kernel_size-1)//2 # padding so that dimension is untampered
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              padding=pad,
                              bias=False,
                             )
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels) # matches out_channels of the Conv layer
        self.spatial_dropout = nn.Dropout2d(p=dropout_rate)  # low rate for CNN layers, like 0.2
        self.pool = None
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # reduces matrix dimension by 2
        
    
    def forward(self, x):
        #print(f"Size before Downsampler:{x.shape}")
        # 1. The Convolutional Layer
        x = self.conv(x)

        # 2. The Activation Function
        x = self.relu(x)

        # 3. Batch Normalization
        x = self.batch_norm(x)       # Normalizes the ReLU activations

        # 4. Spatial Dropout
        x = self.spatial_dropout(x)  # Safely drops channels AFTER normalization

        # 5. Pooling Layer
        if self.pool:
            x = self.pool(x)
        #print(f"Size after Downsampler:{x.shape}")
        return x
