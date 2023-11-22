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