import torch

from model import UNet
from data_helpers import reverse_transform
from sampling import algo_two_simple

model = UNet(channels = 1, dim = 28, dim_mults=[1,2])
model.load_state_dict(torch.load("./2024-07-05-19:23:05.pth"))
model.eval()

# i, n = generate_simple(model)
# im = reverse_transform(i[0].detach())
# print(type(im))
# im.show()

xs = algo_two_simple(model)
for i in [1, 50, 100, 150, 200, 250, 299]:
    im = reverse_transform(xs[i][0].detach())
    im.save(f"run-5-{i}.jpg")

