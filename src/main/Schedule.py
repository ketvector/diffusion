import torch

def linear_beta_schedule(timesteps):
  start = 0.0001
  end = 0.02
  return torch.linspace(start, end, timesteps)


class LinearScheduleHolder():
  
  def __init__(self):
    self.schedule = linear_beta_schedule(300)
    self.timesteps = self.schedule.size(0)
    self.betas = self.schedule
    self.alphas = 1.0 - self.betas
    self.alpha_bars = torch.cumprod(self.alphas, axis=0)
    self.one_minus_alpha_bars = 1 - self.alpha_bars
    self.alpha_bar_roots = torch.sqrt(self.alpha_bars)
    self.one_minus_alpha_bar_roots = torch.sqrt(self.one_minus_alpha_bars)
    
  def add_dims(self, x):
    return x.unsqueeze(-1).unsqueeze(-1)
  
  def alpha(self, t):
    return self.add_dims(self.alphas[t])

  def beta(self,t):
    return self.add_dims(self.betas[t])
  
  def alpha_bar(self,t):
    return self.add_dims(self.alpha_bars[t])
  
  def one_minus_alpha_bar(self,t):
    return self.add_dims(self.one_minus_alpha_bars[t])

  def one_minus_alpha_bar_root(self,t):
    return self.add_dims(self.one_minus_alpha_bar_roots[t])

  def alpha_bar_root(self,t):
    return self.add_dims(self.alpha_bar_roots[t])
  
def get_noisy_image_sample(x_0, t):
  noise = torch.randn_like(x_0)
  s = LinearScheduleHolder()
  return (x_0 * s.alpha_bar_root(t) + noise * s.one_minus_alpha_bar_root(t), noise)