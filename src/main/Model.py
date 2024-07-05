import torch
from torch import nn
from ResNetBlock import ResNetBlock
from NetworkPieces import DownSample, UpSample
from SinusoidalPositionEmbedding import SinusoidalPositionEmbedding

class UNet(nn.Module):
  
  def __init__(self, channels = 1, dim = 28, dim_mults=[1,2]):
    super().__init__()
    self.dim = dim
    self.channels = channels
    self.init_conv = nn.Conv2d(self.channels, dim, 1, padding=0)
    self.downs = nn.ModuleList([])
    self.ups = nn.ModuleList([])
    self.in_out = [(dim, dim)] + [(dim * dim_mults[i] , dim * dim_mults[i+1]) for i in range(len(dim_mults)-1)]
    for ind, (dim_in, dim_out) in enumerate(self.in_out):
      is_last = (ind == len(self.in_out)-1) 
      self.downs.append(
          nn.ModuleList([
              ResNetBlock(dim_in, dim_in),
              ResNetBlock(dim_in,dim_in),
              DownSample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
          ])
      )
    
    mid_dim = dim * dim_mults[-1]
    self.mid_block1 = ResNetBlock(mid_dim, mid_dim)
    for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out)):
      is_last = (ind == len(self.in_out)-1) 
      self.ups.append(
          nn.ModuleList([
              ResNetBlock(dim_out + dim_in, dim_out),
              ResNetBlock(dim_out + dim_in, dim_out),
              UpSample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
          ])
      )
    self.final_conv = nn.Conv2d(dim, self.channels, 1)

  def forward(self,x,t):
      t_embedding = ((SinusoidalPositionEmbedding(self.dim * self.dim)(t)).view(t.size(0), self.dim, self.dim)).unsqueeze(1)
      x = x + t_embedding
      x = self.init_conv(x)
      h = []
      #print(f"self.downs : {self.downs}")
      for block1, block2, downsample in self.downs:
        x = block1(x)
        h.append(x)
        x = block2(x)
        h.append(x)
        x = downsample(x)
      
      x = self.mid_block1(x)


      for block1, block2, upsample in self.ups:
        x = block1(torch.cat((h.pop(), x), dim = 1))
        x = block2(torch.cat((h.pop(), x), dim = 1))
        x = upsample(x)
      
      x = self.final_conv(x)
      return x

def get_loss(actual, prediction):
  return nn.MSELoss()(actual, prediction)


