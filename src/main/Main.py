
import torch
from Model import get_loss, UNet
from Schedule import get_noisy_image_sample
from Data import get_train_data_loader, reverse_transform
from time import time
from datetime import datetime

def generate_simple(model):
  final_sample = torch.randn(size=(1,1,28,28))
  predicted_noise = model(final_sample, torch.tensor([[299]]))
  image_raw = final_sample - predicted_noise
  return image_raw, predicted_noise

def train(model, optimizer, train_dataloader):
  timesteps = 300
  epochs = 10
  current_time_pair = None
  for epoch in range(epochs):
    for step, (batch, _) in enumerate(train_dataloader):
      optimizer.zero_grad()
      batch_size = batch.size()[0]
      ts = torch.randint(0, timesteps, (batch_size,1))
      noisy_samples, noise = get_noisy_image_sample(batch, ts)
      loss = get_loss(noise, model(noisy_samples,ts))
      
      if step % 100 == 0:
        if not current_time_pair:
          current_time_pair = (time(), time())
        else:
          current_time_pair = (current_time_pair[1], time())
        print(f"epoch: {epoch} , step: {step},  loss {loss.item()}, last lap: {current_time_pair[1] - current_time_pair[0]}")
        
      
      loss.backward()

      optimizer.step()

def main():
    model = UNet(channels = 1, dim = 28, dim_mults=[1,2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data_loader = get_train_data_loader()
    train(model, optimizer, train_data_loader)
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d-%H:%M:%S')
    torch.save(model.state_dict(), f"./{formatted_time}")
    model.eval()
    image_raw, predicted_noise = generate_simple(model)
    reverse_transform(image_raw[0].detach())

#main()