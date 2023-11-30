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
        self.layerNorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.layerNorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.patchify = lambda x: window_partition(x, window_size)
        self.unpatchify = lambda x, H, W: window_reverse(x, window_size, H, W)
    
    def forward(self, x, pos_embed):
        _, _, H, W = x.shape

        x = self.resBlock(x)
        x = self.patchify(x) + pos_embed
        x = x + self.layerNorm1(self.attn(x))
        x = x + self.layerNorm2(self.mlp(x))
        
        return self.unpatchify(x, H, W)
        # return x


class betaChessAI(nn.Module):
    def __init__(self, 
                 depth=10, 
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

        self.linear1 = nn.Linear(in_features=hidden_channel * 64, out_features=64 * 64, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_channel * 64, out_features=5, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, actions, side):
        B, _, _ = x.shape

        x = encodeBoard(x, side, B).to(self.device) # (B, 13, 8, 8)
        x = self.conv1(x) # (B, hidden_channel, 8, 8)

        # stack = []

        for block in self.blocks:
            x = block(x, self.pos_embed) # (B, hidden_channel, 8, 8)
        # for i in range(len(self.blocks)):
        #     if i < 5:
        #         x = self.blocks[i](x, self.pos_embed) # (B, hidden_channel, 8, 8)
        #         stack.append(x)
        #     else:
        #         z = stack.pop()
        #         x = z + self.blocks[i](x, self.pos_embed) # (B, hidden_channel, 8, 8)

        special_actions = self.linear2(rearrange(x, "B C H W -> B (H W C)"))
        x = self.linear1(rearrange(x, "B C H W -> B (H W C)"))

        all_actions = torch.cat((x, special_actions), dim=1)

        return self.softmax(all_actions)
        # return decodeOutput(x, special_actions, B, mask).to(self.device) # (B, 8 * 8 * 64 + 5)

class valueNet(nn.Module):
    def __init__(self, 
                 depth=10, 
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
            x = block(x, self.pos_embed) # (B, hidden_channel, 8, 8)
    
        x = self.mlp(rearrange(x, "B C H W -> B (H W C)"))

        return x