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
                                        transform=torchvision.transforms.ToTensor(),

                                        )
test_data = torchvision.datasets.MNIST(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
for i in range(1, 11):
    img = np.transpose(train_data[i][0], (1, 2, 0))
    plt.subplot(2, 10, i)
    plt.imshow(img*255)

train_size = len(train_data)
test_size = len(test_data)

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=128, sampler=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            # 对图片进行编码
            # ---------------------------------------
            nn.Conv2d(1, 32, (3, 3), padding='same'),
            nn.ReLU(inplace=False),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.MaxPool2d((2, 2), padding=0),
            # -------------------------------------
        )
        self.decoder=nn.Sequential(
            # 对图像解码
            # ---------------------------------------
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(32, 1, (3, 3), padding='same')
            # -------------------------------------
        )

    def forward(self, x):
        x = self.encoder(x)
        x=self.decoder(x)
        return x


# class loss_f(nn.Module):
#     def __init__(self):
#         super(loss_f, self).__init__()


# 给图片添加高斯噪声
def noise(imgs):
    empty=[]
    for img,target in imgs:
        noise = torch.randn(img.size())
        img_noise = noise*noise_ratio+ img*(1-noise_ratio)
        img_noise=img_noise.clip(0,1)
        empty.append((img,img_noise))
    return empty

def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))



img_and_img_noise = noise(train_dataloader)
train_data_noise = train_dataloader
train_data_noise.dataset.data = img_and_img_noise

for i in range(1, 11):
    img_noise_show = train_data_noise.dataset.data[0][1]
    img_noise_show=img_noise_show[i]
    plt.subplot(2, 10, i + 10)
    plt.imshow(np.transpose(img_noise_show, (1, 2, 0)))
plt.ion()
plt.figure(1)
plt.show()
plt.pause(0.1)

print(1)
# model=torch.load("my_model1600.pth")
model = Net()
loss_fn = nn.MSELoss()
learn_rate=0.00001
optimizer = optim.Adam(model.parameters(),lr=learn_rate)

epoch = 3000
for i in range(epoch):
    for data in train_data_noise.dataset.data:
        imgs, noise = data
        # plt.figure(2)
        # plt.subplot(1,3,1)
        # plt.imshow(np.transpose(imgs[0],(1,2,0)))
        # plt.subplot(1,3,2)
        # plt.imshow(np.transpose(noise[0],(1,2,0)))
        imgs=imgs
        noise=noise
        output = model(noise)
        # output_c=output.cpu()
        # output_c=output_c.detach()
        # plt.subplot(1,3,3)
        # plt.imshow(np.transpose(output_c[0],(1,2,0)))
        # plt.show()

        loss = loss_fn(output, imgs)*1000

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i % 100 == 0:
        the_current_time()
        print("第{}轮训练损失:".format(i))
        print(loss.item())
        torch.save(model,"my_model{}.pth".format(i))
        print("第{}轮模型已保存:".format(i))

