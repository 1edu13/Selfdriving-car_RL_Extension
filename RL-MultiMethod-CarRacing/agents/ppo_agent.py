"""
PPO Agent for CarRacing-v2
Proximal Policy Optimization with continuous action space.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

from core.cnn_backbone import CNNBackbone

