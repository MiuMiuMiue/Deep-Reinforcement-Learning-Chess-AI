from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

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

def encodeBoard(x, side, B):
    assert x.shape == (B, 8, 8)

    B, _, _ = x.shape
    boards = np.zeros((B, 13, 8, 8))
    
    for batch in range(B):
        for piece in range(1, 7):
            boards[batch][piece - 1, :, :] = x[batch] == piece

        for piece in range(1, 7):
            boards[batch][piece + 5, :, :] = x[batch] == -1 * piece

        boards[batch][12, :, :] = side[batch]

    return torch.tensor(boards).float()

def decodeOutput(x, y, B):
    assert x.shape == (B, 64, 8, 8)
    assert y.shape == (B, 5)

    x = x
    sm = nn.Softmax(dim=1)

    x = rearrange(x, "B C H W -> B (H W C)")
    all_actions = torch.cat((x, y), dim=1)

    all_actions = sm(all_actions)
    # all_actions = torch.argmax(all_actions, axis=1)
    return all_actions # (B, 64 * 8 * 8 + 5)

actionDict = {"CASTLE_KING_SIDE_WHITE": 0, 
              "CASTLE_QUEEN_SIDE_WHITE": 1, 
              "CASTLE_KING_SIDE_BLACK": 2, 
              "CASTLE_QUEEN_SIDE_BLACK": 3, 
              "RESIGN": 4}

def action_to_move(action):
    if action >= 64 * 64:
        _action = action - 64 * 64
        if _action == 0:
            return "CASTLE_KING_SIDE_WHITE"
        elif _action == 1:
            return "CASTLE_QUEEN_SIDE_WHITE"
        elif _action == 2:
            return "CASTLE_KING_SIDE_BLACK"
        elif _action == 3:
            return "CASTLE_QUEEN_SIDE_BLACK"
        elif _action == 4:
            return "RESIGN"

    _from, _to = action // 64, action % 64
    x0, y0 = _from // 8, _from % 8
    x1, y1 = _to // 8, _to % 8
    return x1 * 8 + y1, x0, y0

def computeMask(legal_actions):
    B, _ = legal_actions.shape
    mask1 = np.zeros((B, 64, 8, 8))
    mask2 = np.zeros((B, 5))

    for i in range(B):
        for action in legal_actions[i]:
            ind = action_to_move(action)
            if ind in actionDict.keys():
                mask2[i, actionDict[ind]] = 1
            else:
                mask1[i, ind[0], ind[1], ind[2]] = 1
    
    return torch.tensor(mask1), torch.tensor(mask2)

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