import torch
import numpy as np
from hc import hc_mine
from hb import hb_mine

def causal_entropy(samples, i, j, K):
    samples = torch.tensor(samples)
    input = [torch.unsqueeze(samples[:, i], dim=1), torch.unsqueeze(samples[:, j], dim=1), samples[:, K]]
    return hc_mine(input)


def mci_test(samples, i, j, K, alpha=0.05):
    
    if len(K) == 0:
        mi = hb_mine(np.expand_dims(samples[:, i], axis=1), np.expand_dims(samples[:, j], axis=1))
        if mi < alpha:
            return True, mi
        else:
            return False, mi

    ce = causal_entropy(samples, i, j, K)
    if ce < alpha:
        return True, ce
    else:
        return False, ce
