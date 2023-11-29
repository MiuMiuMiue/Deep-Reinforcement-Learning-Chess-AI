from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
import random
from env import ChessEnvV1
import traceback

TESTING = False

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
    x = x.cpu().numpy()
    side = side.cpu().numpy()

    B, _, _ = x.shape
    boards = np.zeros((B, 13, 8, 8))
    
    for batch in range(B):
        for piece in range(1, 7):
            boards[batch][piece - 1, :, :] = x[batch] == piece

        for piece in range(1, 7):
            boards[batch][piece + 5, :, :] = x[batch] == -1 * piece

        boards[batch][12, :, :] = side[batch]

    return torch.tensor(boards).float()

def decodeOutput(x, y, B, mask):
    assert x.shape == (B, 64 * 8 * 8)
    assert y.shape == (B, 5)
    global TESTING

    sm = nn.Softmax()
    all_actions = torch.cat((x, y), dim=1) * mask
    if TESTING:
        print(all_actions)
    
    for i in range(B):
        action_mask = all_actions[i] != 0
        probs = sm(all_actions[i, action_mask])
        all_actions[i, action_mask] = probs

    return all_actions # (B, 64 * 8 * 8 + 5)

def computeMask(legal_actions):
    B, _ = legal_actions.shape
    legal_actions = legal_actions.cpu().numpy()
    mask = np.zeros((B, 4101))

    for i in range(B):
        mask[i, legal_actions[i][legal_actions[i] != -1]] = 1
    return torch.tensor(mask)

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

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def save_ckpt(policyModel, valueModel, policyOptim, valueOptim, dir, step):
    checkpoint = {
        "policyModel": policyModel.state_dict(), 
        "valueModel": valueModel.state_dict(), 
        "policyOptim": policyOptim.state_dict(), 
        "valueOptim": valueOptim.state_dict()
        }
    ckpt_path = os.path.join(dir, f"{step:07d}.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"\tcheckpoint saved to {ckpt_path}")

def resume_from_ckpt(policyModel, valueModel, policyOptim, valueOptim, ckptPath):
    assert os.path.isfile(ckptPath), f"could not find model checkpoint at {ckptPath}"
    print("Loading Checkpoints ...")
    checkpoints = torch.load(ckptPath, map_location=lambda storage, loc: storage)
    policy_model = checkpoints["policyModel"]
    value_model = checkpoints["valueModel"]
    policy_optim = checkpoints["policyOptim"]
    value_optim = checkpoints["valueOptim"]

    policyModel.load_state_dict(policy_model, strict=True)
    valueModel.load_state_dict(value_model, strict=True)
    policyOptim.load_state_dict(policy_optim)
    valueOptim.load_state_dict(value_optim)
    print("Finish Loading")
    return policyModel, valueModel, policyOptim, valueOptim

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def switchTeacherStudent(student, teacher, device):
    global TESTING
    TESTING = True
    game_env = ChessEnvV1(device=device, log=False)
    with torch.no_grad():
        count = 0
        for _ in range(10):
            print("\tgame starts ...")
            side = random.choice((0, 1))
            state = game_env.reset(player_color="WHITE", opponent=teacher) if side == 0 else game_env.reset(player_color="BLACK", opponent=teacher)
            print(f"\tside info: {side}")
            done = False
            while not done:
                moveCount = game_env.move_count
                actions = game_env.possible_actions

                action_probs = student(torch.tensor([state]).to(device), torch.tensor([actions]).to(device), torch.tensor([side]).to(device))
                action = torch.argmax(action_probs[0]).item()
                new_state, reward, done, info = game_env.step(action)

                state = new_state
                if moveCount == game_env.move_count:
                    print(action_probs)
                    print(action_probs[0, torch.tensor(actions)])
                    print(action_probs.max())
                    print(torch.argmax(action_probs))
                    print(actions)
                    print("==============================================")

            print(f"\tgame result reward: {reward}")
            if reward >= 0:
                count += 1
    
    return count

    