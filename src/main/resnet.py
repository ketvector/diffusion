from torch import nn

class Block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.proj = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm = nn.BatchNorm2d(out_channels)
    self.act = nn.SiLU()

  def forward(self, x):
    return self.act(self.norm(self.proj(x)))
  

class ResNetBlock(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.block1 = Block(dim_in, dim_out)
    self.block2 = Block(dim_out, dim_out)
    self.res_conv = nn.Conv2d(dim_in, dim_out, 1)

  def forward(self,x):
     y1 = self.block1(x)
     h = self.block2(y1)
     return self.res_conv(x) + h