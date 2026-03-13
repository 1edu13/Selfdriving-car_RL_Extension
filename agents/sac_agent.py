"""
SAC Agent for CarRacing-v2.
Soft Actor-Critic with automatic entropy tuning.
"""

import copy
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from core.cnn_backbone import CNNBackbone
