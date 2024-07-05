import torch
from main.SinusoidalPositionEmbedding import SinusoidalPositionEmbedding 
import matplotlib.pyplot as plt 

t = torch.arange(50).view(50,1)
s = SinusoidalPositionEmbedding(128)
v = s(t)


plt.imshow(v)
plt.show()