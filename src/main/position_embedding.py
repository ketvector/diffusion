import torch
from torch import nn
import math


class SinusoidalPositionEmbedding(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim
  def forward(self, position):
    base = -1.0 * (math.log(10000.0) / self.embedding_dim)
    single_position_base = torch.arange(0, self.embedding_dim,2)
    omegas = torch.exp(single_position_base * base).unsqueeze(0)
    omega_ts = omegas * position
    sines = omega_ts.sin()
    cosines = omega_ts.cos()
    pes = torch.zeros(position.size()[0], self.embedding_dim)
    pes[:, 0::2] = sines
    pes[:, 1::2] = cosines
    return pes