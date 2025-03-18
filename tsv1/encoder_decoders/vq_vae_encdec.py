"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import timefreq_to_time, time_to_timefreq, SnakeActivation


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, frequency_indepence:bool, mid_channels=None, dropout:float=0.):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels
        
        kernel_size = (1, 3) if frequency_indepence else (3, 3)
        padding = (0, 1) if frequency_indepence else (1, 1)

        layers = [
            SnakeActivation(in_channels, 2), #SnakyGELU(in_channels, 2), #SnakeActivation(in_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.Dropout(dropout)
        ]
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x) + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frequency_indepence:bool,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEDecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frequency_indepence:bool,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 init_dim:int,
                 hid_dim: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 pad_func,
                 n_fft:int,
                 frequency_indepence:bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param bn: use of BatchNorm
        :param kwargs:
        """
        super().__init__()
        self.pad_func = pad_func
        self.n_fft = n_fft


        d = init_dim
        enc_layers = [VQVAEEncBlock(num_channels, d, frequency_indepence),]
        d *= 2
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            enc_layers.append(VQVAEEncBlock(d//2, d, frequency_indepence))
            for _ in range(n_resnet_blocks):
                enc_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d *= 2
        enc_layers.append(ResBlock(d//2, hid_dim, frequency_indepence, dropout=dropout))
        self.encoder = nn.Sequential(*enc_layers)

        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.tensor(0))
        self.register_buffer('H_prime', torch.tensor(0))
        self.register_buffer('W_prime', torch.tensor(0))
    
    def forward(self, x):
        """
        :param x: (b c l)
        """
        # Print shapes for debugging
        print(f"VQVAEEncoder input shape: {x.shape}, device: {x.device}")
        
        # Get the device of encoder parameters for comparison
        encoder_device = next(self.encoder.parameters()).device
        print(f"Encoder device: {encoder_device}")
        
        in_channels = x.shape[1]
        x = time_to_timefreq(x, self.n_fft, in_channels)  # (b c h w)
        print(f"After time_to_timefreq: {x.shape}, device: {x.device}")
        
        # Make sure x is on the same device as the encoder
        if x.device != encoder_device:
            print(f"Moving tensor from {x.device} to {encoder_device}")
            x = x.to(encoder_device)
        
        # Apply padding function
        x = self.pad_func(x, copy=True)   # (b c h w)
        print(f"After pad_func: {x.shape}, device: {x.device}")

        # Pass through encoder
        try:
            out = self.encoder(x)  # (b c h w)
            print(f"After encoder: {out.shape}")
        except RuntimeError as e:
            # Display helpful error message
            print(f"Error in encoder: {e}")
            print(f"Input shape: {x.shape}, device: {x.device}")
            print(f"Encoder device: {encoder_device}")
            print(f"First encoder layer input channels: {next(self.encoder.parameters()).shape[1]}")
            print(f"Actual input channels: {x.shape[1]}")
            raise
        
        if not self.is_num_tokens_updated:
            self.H_prime = torch.tensor(out.shape[2])
            self.W_prime = torch.tensor(out.shape[3])
            self.num_tokens = self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 init_dim:int,
                 hid_dim: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 input_length:int,
                 pad_func,
                 n_fft:int,
                 x_channels:int,
                 frequency_indepence:bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        self.pad_func = pad_func
        self.n_fft = n_fft
        self.x_channels = x_channels
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)
        
        d = int(init_dim * 2**(int(round(np.log2(downsample_rate))) - 1))  # enc_out_dim == dec_in_dim
        if round(np.log2(downsample_rate)) == 0:
            d = int(init_dim * 2**(int(round(np.log2(downsample_rate)))))
        dec_layers = [ResBlock(hid_dim, d, frequency_indepence, dropout=dropout)]
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            for _ in range(n_resnet_blocks):
                dec_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d //= 2
            dec_layers.append(VQVAEDecBlock(2*d, d, frequency_indepence))
        dec_layers.append(nn.ConvTranspose2d(d, num_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding))
        dec_layers.append(nn.ConvTranspose2d(num_channels, num_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding))
        self.decoder = nn.Sequential(*dec_layers)

        self.interp = nn.Upsample(input_length, mode='linear')
        # self.linear = nn.Linear(input_length, input_length)  # though helpful, it consumes too much memory for long sequences
        

    def forward(self, x):
        out = self.decoder(x)  # (b c h w)
        out = self.pad_func(out)  # (b c h w)
        out = timefreq_to_time(out, self.n_fft, self.x_channels)  # (b c l)

        out = self.interp(out)  # (b c l)
        # out = out + self.linear(out)  # (b c l)
        return out
