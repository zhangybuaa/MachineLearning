import numpy as np
import torchvision
from torchvision import datasets, transforms
import os
import cv2

# ims_path='/home/BH/zy1906304/zy/deeplearning/cifar-10/data/train/'
# ims_list=os.listdir(ims_path)
#
# R_means=[]
# G_means=[]
# B_means=[]
# for im_list in ims_list:
# 	im=cv2.imread(ims_path+im_list)
# 	im_R = im[:, :, 0]
# 	im_G = im[:, :, 1]
# 	im_B = im[:, :, 2]
# # count mean for every channel
# 	im_R_mean = np.mean(im_R)
# 	im_G_mean = np.mean(im_G)
# 	im_B_mean = np.mean(im_B)
# 	# save single mean value to a set of means
# 	R_means.append(im_R_mean)
# 	G_means.append(im_G_mean)
# 	B_means.append(im_B_mean)
#
# mean=[0,0,0]
# #count the sum of different channel means
# mean[0]=np.mean(R_means)
# mean[1]=np.mean(G_means)
# mean[2]=np.mean(B_means)
# print('数据集的RGR平均值为\n[{}，{}，{}]'.format( mean[0],mean[1],mean[2]) )


data_transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])
trainset = torchvision.datasets.ImageFolder(root='/home/BH/zy1906304/zy/deeplearning/cifar-10/data',
                                            transform=data_transform)


data = [d[0].data.cpu().numpy() for d in trainset]
print(np.mean(data))