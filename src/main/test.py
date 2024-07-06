import torch

from model import UNet
from data_helpers import reverse_transform
from sampling import algo_two_simple
import constants

# TODO: generalise more
def get_sample_steps():
    num_images = constants.NUM_IMAGES_CONCAT_RESULT
    step_size = (constants.TIMESTEPS - 2) // (num_images - 1) 
    sample_steps = [0] + list(range(step_size, constants.TIMESTEPS - step_size, step_size)) + [constants.TIMESTEPS-1]
    return sample_steps

def infer(saved_weights_path, save_as_name):
    model = UNet(channels = 1, dim = 28, dim_mults=[1,2])
    model.load_state_dict(torch.load(saved_weights_path))
    model.eval()
    
    xs = algo_two_simple(model)
    result = None
    
    for i in get_sample_steps():
        if result == None:
            result = xs[i][0].detach()
        else:
            result = torch.cat([result, xs[i][0].detach()], axis = 2)

    im = reverse_transform(result)
    im.save(save_as_name)
    print("saved results ....")

if __name__ == "__main__":
    infer("./2024-07-05-19:23:05.pth", "results.jpg")
    

