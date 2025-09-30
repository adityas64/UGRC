import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
from Models import IF

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict.get(verbosity, logging.INFO))
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', model_only=False):
    if model_only:
        filename = filename.replace('checkpoint', 'model_only_checkpoint')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace('checkpoint', 'model_bests')
        os.makedirs(os.path.dirname(best_filename), exist_ok=True)  
        torch.save(state, best_filename)
