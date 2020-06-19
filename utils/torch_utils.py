"""
Created on 2020/6/11 18:39 周四
@author: Matt zhuhan1401@126.com
Description: description
"""

import random
import numpy as np

import torch

def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)