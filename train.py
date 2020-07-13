#@title Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import Descriminator, Generator

#@title Downlaod dataset (run this cell only once)
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(64), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='.', download=True, train=True, transform=transforms)
testset = torchvision.datasets.MNIST(root='.', download=True, train=False, transform=transforms)

#@title Loading dataset
train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)

for batch in train:
    print(batch)
    break

images, labels = batch
print(images.shape)

plt.figure(figsize=(10, 10))

for i in range(len(images)):
    plt.subplot(8, 8, i+1)
    plt.imshow(images[i].squeeze())
    plt.ylabel(labels[i].item(), color='white')
    plt.xticks([]), plt.yticks([])

plt.show()

batch_size = 64
num_channels = batch[0][0].shape[0]
img_size = batch[0][0].shape[1]
channel_noise = 256
lr = 0.0005
features_d = 16
features_g = 16
num_epoch = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(batch_size, num_channels, img_size, img_size)

descr = Descriminator(num_channels, features_d).to(device)
gen = Generator(num_channels, features_g, channel_noise).to(device)
optim_d = torch.optim.Adam(descr.parameters(), lr=lr, betas=(0.5, 0.999))
optim_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

descr.train()
gen.train()

label_real = 1
label_fake = 0
initial_noise = torch.randn(64, channel_noise, 1, 1).to(device)
writer_for_real = SummaryWriter('logs/gan/real')
writer_for_fake = SummaryWriter('logs/gan/fake')

print(f'training...')
for epoch in range(1, num_epoch+1):

    for i, data in enumerate(train):

        images, target = data

        images = images.to(device)
        batch_size = images.shape[0]

        ### training descriminator
        descr.zero_grad()
        labels_real = (torch.ones(batch_size)* 0.9).to(device)
        output_d = descr(images).reshape(-1)
        loss_d_real = criterion(output_d, labels_real)

        temp_noise = torch.randn(batch_size, channel_noise, 1, 1).to(device)
        fake_images = gen(temp_noise)
        labels_fake = (torch.ones(batch_size)*0.1).to(device)
        output_d = descr(fake_images.detach()).reshape(-1)
        loss_d_fake = criterion(output_d, labels_fake)


        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optim_d.step()
        

        ### training generator
        gen.zero_grad()
        labels = torch.ones(batch_size).to(device)
        output = descr(fake_images).reshape(-1)
        loss_g = criterion(output, labels)

        loss_g.backward()
        optim_g.step()


        ### 
        if i % 100  == 0:

            print(
                f'epoch : {epoch}/{num_epoch}, batch_idx : {i} , loss_d : {loss_d}, \
                loss_g : {loss_g}'
            )
            
            with torch.no_grad():
                fake_images = gen(initial_noise)
                grid_real = torchvision.utils.make_grid(images[:32], normalize=True)
                grid_fake = torchvision.utils.make_grid(fake_images[:32], normalize=True)
                writer_for_real.add_image('Real Images', grid_real)
                writer_for_fake.add_image('Real Images', grid_fake)