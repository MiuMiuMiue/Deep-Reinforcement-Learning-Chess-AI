import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from utils import *
import numpy as np
from einops import rearrange

def window_partition(x, window_size):
    # This is the Patchify function that can divide the image into smaller patch
    # The output size with default value of reshape_seq is (B, H // window_size, W // window_size, window_size, window_size, C)
    # Batch size, number of window in height, number of window in width, window size, window size, channel

    # reshape_seq = True: Reshape the 2D patch into a 1D sequence
    # The output size with reshape_seq = True is (B, window_size * window_size, C) (H, W, C)
    # Batch size, Patch Sequence, Channal

    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1)
    windows = rearrange(windows, 'b h w p1 p2 c -> b (h w) (p1 p2 c)')
    return windows # (B, N, P^2*C)

def window_reverse(windows, window_size, H, W):
    # Reverse the patchified image back to the original image
    # (B, N, P^2*C)
    windows = rearrange(windows, 'b (h w) (p1 p2 c) -> b h w p1 p2 c', p1=window_size, p2=window_size, h=H // window_size, w =W // window_size)
    windows = rearrange(windows, 'b h w p1 p2 c -> b c (h p1) (w p2)')
    return windows # (B C H W)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def encodeBoard(x, B):
    assert x.shape == (B, 128)
    x = x.reshape((B, 2, 8, 8))
    boards = torch.zeros((B, 12, 8, 8))
    
    for batch in range(B):
        for piece in range(1, 7):
            boards[batch][piece - 1, :, :] = x[batch, 0] == piece

        for piece in range(1, 7):
            boards[batch][piece + 5, :, :] = x[batch, 1] == piece

    return torch.tensor(boards).float()

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

        self.mlp = Mlp(in_features=hidden_size, hidden_features=hidden_size, act_layer=nn.ReLU)
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
                 hidden_channel=64,
                 num_heads=8,
                 window_size=2,
                 input_size=8, 
                 mlp_ratio=4.0):
        super(betaChessAI, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_patches = (input_size // window_size) ** 2

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            betaChessBlock(hidden_channel * window_size ** 2, hidden_channel, num_heads, window_size, mlp_ratio) for _ in range(depth)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, int((input_size / window_size) ** 2), hidden_channel * window_size ** 2), requires_grad=False)

        self.linear1 = nn.Linear(in_features=hidden_channel * 64, out_features=1024, bias=True)
        self.linear2 = nn.Linear(in_features=1024, out_features=640, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.initialize_weights()
        self.to(self.device)
    
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        B, D = x.shape

        x = encodeBoard(x, B).to(self.device) # (B, 12, 8, 8)
        x = self.conv1(x) # (B, hidden_channel, 8, 8)

        for block in self.blocks:
            x = block(x, self.pos_embed) # (B, hidden_channel, 8, 8)

        x = self.linear2(self.relu(self.linear1(rearrange(x, "B C H W -> B (H W C)"))))

        return self.softmax(x)

class betaChessValue(nn.Module):
    def __init__(self, 
                 depth=10, 
                 hidden_channel=64,
                 num_heads=8,
                 window_size=2,
                 input_size=8, 
                 mlp_ratio=4.0):
        super(betaChessValue, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_patches = (input_size // window_size) ** 2

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=hidden_channel, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            betaChessBlock(hidden_channel * window_size ** 2, hidden_channel, num_heads, window_size, mlp_ratio) for _ in range(depth)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, int((input_size / window_size) ** 2), hidden_channel * window_size ** 2), requires_grad=False)

        self.linear1 = nn.Linear(in_features=hidden_channel * 64, out_features=1024, bias=True)
        self.linear2 = nn.Linear(in_features=1024, out_features=1, bias=True)
        self.relu = nn.ReLU()

        self.initialize_weights()
        self.to(self.device)
    
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        B, D = x.shape

        x = encodeBoard(x, B).to(self.device) # (B, 12, 8, 8)
        x = self.conv1(x) # (B, hidden_channel, 8, 8)

        for block in self.blocks:
            x = block(x, self.pos_embed) # (B, hidden_channel, 8, 8)

        x = self.linear2(self.relu(self.linear1(rearrange(x, "B C H W -> B (H W C)"))))

        return x