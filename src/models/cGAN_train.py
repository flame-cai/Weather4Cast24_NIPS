import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd
import h5py
import time
from src.utils.metrics import crps

# Custom Dataset
class Pix2PixTestDataset(Dataset):
    def __init__(self, input_sequence_length=1, output_sequence_length=1):
        self.input_data = '../'#path to input
        self.target_data = '../'#path to target
        self.hrit_file = h5py.File(self.input_data, 'r')
        self.opera_file = h5py.File(self.target_data, 'r')
        # valid_indices = pd.read_csv('../')['valid_indices'].tolist()
        self.input_data = self.hrit_file['Binarized-REFL-BT'][:]/150 -1
        self.target_data = self.opera_file['rates.crop'][:]
        self.target_data = self.target_data/5 -1
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.input_data = np.nan_to_num(self.input_data, nan=np.nanmax(self.input_data), posinf=np.nanmax(self.input_data), neginf=np.nanmax(self.input_data))
        self.target_data = np.nan_to_num(self.target_data, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.input_data) - self.input_sequence_length - self.output_sequence_length + 1

    def pad_to_256(self, tensor):
        return F.pad(tensor, (0, 256 - tensor.shape[-1], 0, 256 - tensor.shape[-2]))

    def __getitem__(self, idx):
        input_sequence = self.input_data[idx:idx + 1]
        target_sequence = self.target_data[idx:idx+1]
        
        # Convert to torch tensors and add channel dimension
        input_sequence = torch.from_numpy(input_sequence).float().unsqueeze(1)
        target_sequence = torch.from_numpy(target_sequence).float().unsqueeze(1)
        
        # Pad to 256x256
        input_sequence = self.pad_to_256(input_sequence)
        target_sequence = self.pad_to_256(target_sequence)
        #print("NAN",torch.isnan(input_sequence).any(), torch.isnan(target_sequence).any())
        #print("INF",torch.isinf(input_sequence).any(), torch.isinf(target_sequence).any())
        return input_sequence, target_sequence

    def __del__(self):
        self.hrit_file.close()
        self.opera_file.close()


# Improved Generator
class ImprovedGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(ImprovedGenerator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, padding=3),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.down1 = self.down_block(features, features*2)
        self.down2 = self.down_block(features*2, features*4)
        self.down3 = self.down_block(features*4, features*8)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*8, features*8, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up3 = self.up_block(features*16, features*4)
        self.up2 = self.up_block(features*8, features*2)
        self.up1 = self.up_block(features*4, features)
        
        self.final = nn.Sequential(
            nn.Conv2d(features*2, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def down_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def up_block(self, in_features, out_features):
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        
        bottleneck = self.bottleneck(d4)
        
        u1 = self.up3(torch.cat([bottleneck, d4], dim=1))
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up1(torch.cat([u2, d2], dim=1))
        
        return self.final(torch.cat([u3, d1], dim=1))

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):  # 1 channel for input + 3 channels for generated/target
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_input):
        # img_A is the input image (1 channel)
        # img_B is either the real target or the generated image (3 channels)
        #img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        # Convert 1-channel to 3-channel
        generated = generated.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        return F.mse_loss(gen_features, target_features)

def train_improved_pix2pix(generator, discriminator, train_loader, num_epochs, device):
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixelwise = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)

    start_time = time.time()
    total_steps = len(train_loader)

    for epoch in range(num_epochs):
        total_g_loss = 0
        total_d_loss = 0
        for i, (real_A, real_B) in enumerate(train_loader):
            epoch_start_time = time.time()

            real_A, real_B = real_A.to(device), real_B.to(device)
            
            # real_A shape: (batch_size, 4, height, width)
            # real_B shape: (batch_size, 16, height, width)

            # Train Discriminator
            optimizer_d.zero_grad()
            
            fake_B = generator(real_A.squeeze(2))
            fake_AB = torch.cat((real_A.squeeze(2), fake_B), 1)  # Concatenate along channel dimension
            pred_fake = discriminator(fake_AB.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            real_AB = torch.cat((real_A.squeeze(2), real_B.squeeze(2)), 1)  # Concatenate along channel dimension
            pred_real = discriminator(real_AB)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            
            fake_B = generator(real_A.squeeze(2))
            fake_AB = torch.cat((real_A.squeeze(2), fake_B), 1)  # Concatenate along channel dimension
            pred_fake = discriminator(fake_AB)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            
            loss_g_pixel = criterion_pixelwise(fake_B, real_B.squeeze(2)) * 100
            
            # Adjust perceptual loss for multi-channel input
            loss_g_perceptual = criterion_perceptual(fake_B,real_B.squeeze(2)) * 10
            
            loss_g = loss_g_gan + loss_g_pixel + loss_g_perceptual
            loss_g.backward()
            optimizer_g.step()

            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()

            # Print progress
            if i % 100 == 0:
                elapsed_time = time.time() - start_time
                elapsed_time_epoch = time.time() - epoch_start_time
                remaining_time_epoch = elapsed_time_epoch / (i + 1) * (total_steps - i - 1)
                
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{total_steps}] "
                      f"D_loss: {loss_d.item():.4f}, G_loss: {loss_g.item():.4f}, "
                      f"Pixel_loss: {loss_g_pixel.item():.4f}, Perceptual_loss: {loss_g_perceptual.item():.4f}")
                print(f"Elapsed time: {elapsed_time:.2f}s, "
                      f"Estimated time remaining for epoch: {remaining_time_epoch:.2f}s")

        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        scheduler_g.step(avg_g_loss)
        scheduler_d.step(avg_d_loss)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s")
        print(f"Average D_loss: {avg_d_loss:.4f}, Average G_loss: {avg_g_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    return generator

# Main execution
if __name__ == "__main__":
    input_sequence_length = 1
    output_sequence_length = 1
    batch_size = 16
    num_epochs = 200
    best_loss = float('inf')

    # Initialize dataset and dataloader
    dataset = Pix2PixTestDataset(input_sequence_length, output_sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4)

    # Initialize generator and discriminator
    generator = ImprovedGenerator(in_channels=input_sequence_length, out_channels=output_sequence_length)
    discriminator = Discriminator(in_channels=input_sequence_length + output_sequence_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.MSELoss()

    # Train the model
    trained_generator = train_improved_pix2pix(generator, discriminator, dataloader, num_epochs, device)
    
    # Save the trained generator
    torch.save(trained_generator.state_dict(), 'improved_generator_combined_200i.pth')
    print("Model saved successfully!")


