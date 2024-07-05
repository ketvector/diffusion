from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import ToPILImage

transform = Compose([ToTensor(), Lambda(lambda x: (2 * x) - 1)])

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

def get_train_data_loader():
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    return train_dataloader