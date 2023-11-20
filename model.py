import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from utils import *
import numpy as np

class resBlock(nn.Module):
    def __init__(self, hidden_channel=128):
        super(resBlock).__init__()
        self.conv1 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(hidden_channel)
        self.batch_norm2 = nn.BatchNorm2d(hidden_channel)
    
    def forward(self, x):
        z = self.batch_norm1(self.conv1(self.relu(x)))
        z = self.batch_norm2(self.conv2(self.relu(z)))

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
                 hidden_size=1024, 
                 hidden_channel=256,
                 num_heads=12,
                 window_size=2,
                 input_size=8, 
                 mlp_ratio=4.0):
        super(betaChessAI).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, int((input_size / window_size) ** 2), hidden_size), requires_grad=False)
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            betaChessBlock(hidden_size, hidden_channel, num_heads, window_size, mlp_ratio) for _ in range(depth)
        ])

        self.approx_gelu = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=64, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=hidden_channel * 64, out_features=5, bias=True)

    def forward(self, x, actions, side):
        B, _, _ = x.shape

        x = encodeBoard(x, side, B) # (B, 13, 8, 8)
        x = self.conv1(x) # (B, hidden_channel, 8, 8)
        mask1, mask2 = computeMask(actions) # (B, 64, 8, 8), (B, 5)

        for block in self.blocks:
            x = block(x, self.pos_embed) # (B, hidden_channel, 8, 8)
        
        x = rearrange(self.conv2(x)) # (B, 64, 8, 8)
        special_actions = self.approx_gelu(self.linear(rearrange(x, "B C H W -> B (H W C)")))

        x = x * mask1 # (B, 64, 8, 8)
        special_actions = special_actions * mask2 # (B, 5)

        return decodeOutput(x, special_actions, B) # (B, 8 * 8 * 64 + 5)