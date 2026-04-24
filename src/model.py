"""Residual CNN with a policy head and a value head, à la AlphaZero."""

import torch
import torch.nn as nn
import torch.nn.functional as F
