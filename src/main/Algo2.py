import torch
from Schedule import LinearScheduleHolder


timesteps = 300
def algo_two_simple(model):
  s = LinearScheduleHolder()
  x_t = torch.randn((1,28,28))
  xs = []
  for t in reversed(range(timesteps)):
    xs.append(x_t)
    t_tensor = torch.tensor([[t]])
    mul = 1.0 / torch.sqrt(s.alpha(t_tensor))
    z_t = model(x_t, t_tensor)
    num = 1.0 - s.alpha(t_tensor)
    denom = torch.sqrt(1.0 - s.alpha_bar(t_tensor))
    x_t =  mul * (x_t - (num/denom)*z_t)
    
  return xs