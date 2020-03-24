import math
import torchvision
import torch
from torchvision import datasets,transforms

root="/home/BH/zy1906304/zy/deeplearning/Fashion-MNIST/data"

train_transform = transforms.Compose([torchvision.transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=train_transform)
mnist_test = datasets.FashionMNIST(root=root, train=False, download=True, transform=test_transform)

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=200, shuffle=True, num_workers=4)



temp_sum = 0
cnt = 0
for X, y in train_iter:
    if y.shape[0] != 200:
        break   # 最后一个batch不足batch_size,这里就忽略了
    channel_mean = torch.mean(X, dim=(0,2,3))  # 按channel求均值(不过这里只有1个channel)
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += channel_mean[0].item()
dataset_global_mean = temp_sum / cnt
print('整个数据集的像素均值:{}'.format(dataset_global_mean))
# 求整个数据集的标准差
cnt = 0
temp_sum = 0
for X, y in train_iter:
    if y.shape[0] != 200:
        break   # 最后一个batch不足batch_size,这里就忽略了
    residual = (X - dataset_global_mean) ** 2
    channel_var_mean = torch.mean(residual, dim=(0,2,3))
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += math.sqrt(channel_var_mean[0].item())
dataset_global_std = temp_sum / cnt
print('整个数据集的像素标准差:{}'.format(dataset_global_std))