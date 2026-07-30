import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(input.size(0), *self.img_shape)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(input.size(0), -1))

def main():
    # Hyperparameters
    latent_dim = 100
    img_shape = (1, 28, 28)
    batch_size = 64
    epochs = 50
    lr = 0.0002

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize generator and discriminator
    generator = Generator(latent_dim, img_shape)
    discriminator = Discriminator(img_shape)

    # Loss function and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Train Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(imgs.size(0), 1)
            fake_labels = torch.zeros(imgs.size(0), 1)

            # Real images
            outputs = discriminator(imgs)
            d_loss_real = adversarial_loss(outputs, real_labels)
            d_loss_real.backward()

            # Fake images
            z = torch.randn(imgs.size(0), latent_dim)
            gen_imgs = generator(z)
            outputs = discriminator(gen_imgs.detach())
            d_loss_fake = adversarial_loss(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_d.step()

            d_loss = d_loss_real + d_loss_fake

            # Train Generator
            optimizer_g.zero_grad()
            gen_labels = torch.ones(imgs.size(0), 1)
            z = torch.randn(imgs.size(0), latent_dim)
            gen_imgs = generator(z)
            outputs = discriminator(gen_imgs)
            g_loss = adversarial_loss(outputs, gen_labels)
            g_loss.backward()
            optimizer_g.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")
                save_image(gen_imgs.data[:25], f"images/{epoch}_{i}.png", nrow=5, normalize=True)

if __name__ == "__main__":
    main()
