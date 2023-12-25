import time

import matplotlib.pyplot as plt
import torchvision
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import torch

noise_ratio = 0.3
train_data = torchvision.datasets.MNIST(root='../data', train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())


train_size = len(train_data)
test_size = len(test_data)

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            # 对图片进行编码
            # ---------------------------------------
            nn.Conv2d(1, 8, (3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(8, 16, (3, 3), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(16, 32, (3, 3), padding='same'),
            nn.ReLU(inplace=True),
            # -------------------------------------

            # 对图像解码
            # ---------------------------------------
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(32, 1, (3, 3), padding='same')
            # -------------------------------------
        )

    def forward(self, x):
        x = self.model(x)
        return x

def noise(imgs):
    for i in range(imgs.shape[0]):
        img = imgs[i]
        noise = torch.randn(img.size())
        img_noise = noise*noise_ratio + img*(1-noise_ratio)
        imgs[i] = img_noise
    return imgs

model=torch.load("model_2/my_model_2_1600.pth", map_location='cpu')
print(model)


for i in range(1, 11):
    img = np.transpose(train_dataloader.dataset[i][0], (1, 2, 0))
    plt.subplot(3, 10, i)
    plt.imshow(img,cmap='gray')
    img_noise=noise(train_dataloader.dataset[i][0])
    plt.subplot(3, 10, i+10)
    plt.imshow(np.transpose(img_noise,(1,2,0)),cmap='gray')
    img_noise=img_noise.unsqueeze(0)
    img_reduce=model(img_noise)
    img_reduce=img_reduce.squeeze(0)
    plt.subplot(3, 10, i+20)
    plt.imshow(np.transpose(img_reduce.detach(),(1,2,0)),cmap='gray')

plt.show()
print(1)
print(1)