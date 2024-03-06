import torch

import torch.nn.functional as F

from torchvision.models import vgg19

from torch.utils.data import DataLoader

import torch.optim as optim

from model import define_model, define_diffusion  # Assuming these functions are defined as before

from data_load import load_data  # Make sure this imports your dataset correctly

from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

import numpy as np




# Define Perceptual Loss

# Define Perceptual Loss



class PerceptualLoss(torch.nn.Module):

    def __init__(self):

        super(PerceptualLoss, self).__init__()

        self.vgg = vgg19(pretrained=True).features[:16].eval()

        for param in self.vgg.parameters():

            param.requires_grad = False


    def forward(self, generated, target):

        # Check if the images are grayscale (1 channel) and convert them to RGB (3 channels)

        if generated.size(1) == 1:  # Assuming [batch_size, channels, height, width]

            generated = generated.repeat(1, 3, 1, 1)  # Repeat the channel 3 times

        if target.size(1) == 1:

            target = target.repeat(1, 3, 1, 1)


        vgg_generated = self.vgg(generated)

        vgg_target = self.vgg(target)

        return F.l1_loss(vgg_generated, vgg_target)
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def train_model(dataloader, model, diffusion):  # Assuming diffusion is part of your model or training routine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model.train()
    model.to(device)

    criterion = torch.nn.MSELoss()

    perceptual_criterion = PerceptualLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #device = next(model.parameters()).device  # Get model's device
    losses = []  # List to save the losses


    

    for epoch in range(500):  # Adjust number of epochs as necessary

        for source_slice, target_slice in dataloader:

            source_slice, target_slice = source_slice.to(device), target_slice.to(device)

            optimizer.zero_grad()


            # Generate a random timestep for each batch

            timesteps = 1000

            t = torch.randint(0, timesteps, (source_slice.size(0),), device=source_slice.device)

            # The model's forward call now includes the timestep

            reconstructed_slice = model(source_slice, t)


            # Calculate losses

            mse_loss = criterion(reconstructed_slice, target_slice)

            perceptual_loss = perceptual_criterion(reconstructed_slice, target_slice)

            total_loss = mse_loss + perceptual_loss  # You can adjust the weighting of each loss as needed

            losses.append(total_loss.item())  # Save the loss for this batch

            # Backpropagate and update model weights

            total_loss.backward()

            optimizer.step()


        print(f"Epoch {epoch+1}, Loss: {total_loss.item()}")


    # Save the trained model

    torch.save(model.state_dict(), 'model_checkpoint_7_gpu.pth')

    plt.plot(moving_average(losses, n=100))    
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss_plot.png')