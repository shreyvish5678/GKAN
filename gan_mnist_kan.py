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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
full_valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_indices = [i for i, target in enumerate(full_trainset.targets) if target == 0]
val_indices = [i for i, target in enumerate(full_valset.targets) if target == 0]

trainset = Subset(full_trainset, train_indices)
valset = Subset(full_valset, val_indices)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

generator = KAN([100, 256, 28 * 28], final_activation="tanh")
discriminator = KAN([28 * 28, 256, 1], final_activation="sigmoid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
adversarial_loss = nn.BCEWithLogitsLoss()

save_dir = './generated_images'
os.makedirs(save_dir, exist_ok=True)
for epoch in range(5):
    generator.train()
    discriminator.train()
    with tqdm(trainloader) as pbar:
        for i, (images, _) in enumerate(pbar):
            valid = torch.ones((images.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((images.size(0), 1), requires_grad=False).to(device)

            real_images = images.view(-1, 28 * 28).to(device)
            generator_optimizer.zero_grad()

            z = torch.randn((images.size(0), 100)).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            generator_optimizer.step()

            discriminator_optimizer.zero_grad()
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            discriminator_optimizer.step()

    generator.eval()
    discriminator.eval()

    gen_loss = 0
    disc_loss = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(valloader):
            valid = torch.ones((images.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((images.size(0), 1), requires_grad=False).to(device)
            real_images = images.view(-1, 28 * 28).to(device)
            
            z = torch.randn((images.size(0), 100)).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            gen_loss += g_loss.item()
            disc_loss += d_loss.item()
            
            if i == 0: 
                gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
                save_image(gen_imgs.data[:25], f"{save_dir}/epoch_{epoch + 1}.png", nrow=5, normalize=True)

    gen_loss /= len(valloader)
    disc_loss /= len(valloader)

    print(
        f"Epoch {epoch + 1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}"
    )

torch.save(generator.state_dict(), 'kan_gen.pth')
torch.save(discriminator.state_dict(), 'kan_disc.pth')
