from torch import nn
from einops.layers.torch import Rearrange


class DownSample(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.layer = nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim_in * 4, dim_out, 1)
    )
  
  def forward(self,x):
    return self.layer(x)


class UpSample(nn.Module):
   def __init__(self, dim_in, dim_out):
    super().__init__()
    self.layer = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim_in, dim_out, 3, padding=1)
    )
  
   def forward(self,x):
    return self.layer(x)