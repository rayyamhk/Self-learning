import torch
import torch.nn.functional as F
from datetime import datetime

def log(str):
    print("%s: %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str))