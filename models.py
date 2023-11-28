import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from utils import *
import numpy as np

class resBlock(nn.Module):
    def __init__(self, hidden_channel=128):
        super(resBlock, self).__init__()
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
        super(betaChessBlock, self).__init__()
        self.resBlock = resBlock(hidden_channel=hidden_channel)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=hidden_size, act_layer=approx_gelu)

        self.patchify = lambda x: window_partition(x, window_size)
        self.unpatchify = lambda x, H, W: window_reverse(x, window_size, H, W)
    
    def forward(self, x, pos_embed):
        _, _, H, W = x.shape

        x = self.resBlock(x)
        x = self.patchify(x) + pos_embed
        x = self.attn(x)
        x = self.mlp(x)
        
        return self.unpatchify(x, H, W)


class betaChessAI(nn.Module):
    def __init__(self, 
                 depth=5, 
                 hidden_size=1024, 
                 hidden_channel=128,
                 num_heads=8,
                 window_size=2,
                 input_size=8, 
                 mlp_ratio=4.0, 
                 device=None):
        super(betaChessAI, self).__init__()
        self.device = device
        self.num_patches = (input_size // window_size) ** 2

        self.conv1 = nn.Conv2d(in_channels=13, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            betaChessBlock(hidden_channel * window_size ** 2, hidden_channel, num_heads, window_size, mlp_ratio) for _ in range(depth)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, int((input_size / window_size) ** 2), hidden_channel * window_size ** 2), requires_grad=False)

        self.instanceNorm1d_1 = nn.InstanceNorm1d(64 * 64)
        self.instanceNorm1d_2 = nn.InstanceNorm1d(5)
        self.instanceNorm2d = nn.InstanceNorm2d(hidden_channel)

        self.linear1 = nn.Linear(in_features=hidden_channel * 64, out_features=64 * 64, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_channel * 64, out_features=5, bias=True)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, actions, side):
        B, _, _ = x.shape

        x = encodeBoard(x, side, B).to(self.device) # (B, 13, 8, 8)
        x = self.conv1(x) # (B, hidden_channel, 8, 8)
        mask = computeMask(actions).to(self.device) # (B, 64 * 64 + 5)

        for block in self.blocks:
            x = self.instanceNorm2d(block(x, self.pos_embed)) # (B, hidden_channel, 8, 8)

        special_actions = self.linear2(rearrange(x, "B C H W -> B (H W C)"))
        special_actions = self.instanceNorm1d_2(special_actions)
        x = self.instanceNorm1d_1(self.linear1(rearrange(x, "B C H W -> B (H W C)")))

        return decodeOutput(x, special_actions, B, mask).to(self.device) # (B, 8 * 8 * 64 + 5)

class valueNet(nn.Module):
    def __init__(self, 
                 depth=5, 
                 hidden_size=1024, 
                 hidden_channel=128,
                 num_heads=8,
                 window_size=2,
                 input_size=8, 
                 mlp_ratio=4.0, 
                 device=None):
        super(valueNet, self).__init__()
        self.device = device
        self.num_patches = (input_size // window_size) ** 2

        self.conv1 = nn.Conv2d(in_channels=13, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            betaChessBlock(hidden_channel * window_size ** 2, hidden_channel, num_heads, window_size, mlp_ratio) for _ in range(depth)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, int((input_size / window_size) ** 2), hidden_channel * window_size ** 2), requires_grad=False)
        self.instanceNorm2d = nn.InstanceNorm2d(hidden_channel)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_channel * 64, hidden_features=hidden_channel * 64, out_features=1, act_layer=approx_gelu)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, side):
        B, _, _ = x.shape

        x = encodeBoard(x, side, B).to(self.device) # (B, 13, 8, 8)
        x = self.conv1(x) # (B, hidden_channel, 8, 8)

        for block in self.blocks:
            x = self.instanceNorm2d(block(x, self.pos_embed)) # (B, hidden_channel, 8, 8)
    
        x = self.mlp(rearrange(x, "B C H W -> B (H W C)"))

        return x