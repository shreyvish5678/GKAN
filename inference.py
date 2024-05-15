from kan_util import KAN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import os
generator = KAN([100, 256, 28 * 28], final_activation="tanh")
generator.load_state_dict(torch.load('kan_gen.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.eval()
generator.to(device)
z = torch.randn(4, 100).to(device)
gen_imgs = generator(z)
gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
save_image(gen_imgs.data, f"generated.png", nrow=2, normalize=True)