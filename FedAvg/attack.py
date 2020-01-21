import copy
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


def add_gaussian_noise(w, scale):
    w_attacked = copy.deepcopy(w)
    if type(w_attacked) == list:
        for k in range(len(w_attacked)):
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    else:
        for k in w_attacked.keys():
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    return w_attacked


def change_weight(w_attack, w_honest, change_rate=0.5):
    w_result = copy.deepcopy(w_honest)
    device = w_attack[list(w_attack.keys())[0]].device
    for k in w_honest.keys():
        w_h = w_honest[k]
        w_a = w_attack[k]

        assert w_h.shape == w_a.shape

        honest_idx = torch.FloatTensor((np.random.random(w_h.shape) > change_rate).astype(np.float)).to(device)
        attack_idx = torch.ones_like(w_h).to(device) - honest_idx

        weight = honest_idx * w_h + attack_idx * w_a
        w_result[k] = weight

    return w_result



