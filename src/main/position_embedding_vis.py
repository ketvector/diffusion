import torch
import matplotlib.pyplot as plt 

from position_embedding import SinusoidalPositionEmbedding 


t = torch.arange(50).view(50,1)
s = SinusoidalPositionEmbedding(128)
v = s(t)


plt.imshow(v)
plt.show()