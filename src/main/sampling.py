import torch
from schedule import LinearScheduleHolder

def get_noisy_image_sample(x_0, t):
  noise = torch.randn_like(x_0)
  s = LinearScheduleHolder()
  return (x_0 * s.alpha_bar_root(t) + noise * s.one_minus_alpha_bar_root(t), noise)

def generate_simple(model):
  final_sample = torch.randn(size=(1,1,28,28))
  predicted_noise = model(final_sample, torch.tensor([[299]]))
  image_raw = final_sample - predicted_noise
  return image_raw, predicted_noise

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