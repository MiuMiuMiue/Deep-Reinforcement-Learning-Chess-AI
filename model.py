import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from utils import *
import numpy as np

class resBlock(nn.Module):
    def __init__(self, hidden_channel=256):
        super(resBlock).__init__()
        self.conv1 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(hidden_channel)
        self.batch_norm2 = nn.BatchNorm2d(hidden_channel)
    
    def forward(self, x):
        z = self.batchNorm1(self.conv1(self.relu(x)))
        z = self.batchNorm2(self.conv2(self.relu(z)))

        return x + z

class betaChessBlock(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 hidden_channel, 
                 num_heads, 
                 window_size, 
                 mlp_ratio=4.0):
        self.resBlock = resBlock(hidden_channel=hidden_channel)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(mlp_ratio * hidden_size), act_layer=approx_gelu)

        self.patchify = lambda x: window_partition(x, window_size)
        self.unpatchify = lambda x, H, W: window_reverse(x, window_size, H, W)
    
    def forward(self, x, pos_embed):
        _, _, H, W = x.shape

        x = self.resBlock(x)
        x = self.patchify(x) + pos_embed
        x = self.mlp(self.attn(x))
        
        return self.unpatchify(x, H, W)


class betaChessAI(nn.Module):
    def __init__(self, 
                 depth=5, 
                 hidden_size=512, 
                 hidden_channel=128, 
                 num_heads=12, 
                 window_size=2,
                 input_size=8, 
                 mlp_ratio=4.0):
        super(betaChessAI).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, int((input_size / window_size) ** 2), hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            betaChessBlock(hidden_size, hidden_channel, num_heads, window_size, mlp_ratio) for _ in range(depth)
        ])
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(mlp_ratio * hidden_size), out_features=1, act_layer=approx_gelu)

    def encodeInput(x, side):
        B, _, _ = x.shape
        boards = np.zeros((B, 13, 8, 8))
        
        for batch in range(B):
            for piece in range(1, 7):
                boards[batch][piece - 1, :, :] = x[batch] == piece

            for piece in range(1, 7):
                boards[batch][piece + 5, :, :] = x[batch] == -1 * piece

            boards[batch][12, :, :] = side[batch]

        return boards