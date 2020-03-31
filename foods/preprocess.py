import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
data_dir = '/home/BH/zy1906304/zy/deeplearning/foods/'

df = pd.read_csv(os.path.join(data_dir,'data/train.csv'))
# print(df.info)



# labels_set = set(labels_np) #获取标签中的种类的数量
file = pd.Series(df['filename']).values
# file = [str(i)+".jpg" for i in file]
# file = [os.path.join(data_dir+'data', i )for i in file ]

train_label = pd.Series (df['label']).values #把数据转成ndarry的格式

X = file[:]
# val_file = file[5000:]
y = pd.Series (df['label']).values #把数据转成ndarry的格式
# train_label= labels[:5000]
# val_label = labels[5000:]
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits,random_state=250,shuffle=True)
i = 1
for train_index, test_index in skf.split(X,y):
	split_train = pd.DataFrame()
	split_train['filename'], split_train['label'] = X[train_index], y[train_index]
	split_train.to_csv("./data/k_folds/train_{}.csv".format(i), index=False)

	split_test = pd.DataFrame()
	split_test['filename'], split_test['label'] = X[test_index], y[test_index]
	split_test.to_csv("./data/k_folds/test_{}.csv".format(i), index=False)
	i += 1


# def default_loader(path):
#     img_pil =  Image.open(path)
#     img_pil = img_pil.resize((224,224))
#     img_tensor = transforms.ToTensor(img_pil)
#     return img_tensor
#
#
# class trainset(Dataset):
#     def __init__(self, loader=default_loader):
#         # 定义好 image 的路径
#         self.images = train_file
#         self.target = train_label
#         self.loader = loader
#
#     def __getitem__(self, index):
#         fn = self.images[index]
#         img = self.loader(fn)
#         target = self.target[index]
#         return img, target
#
#     def __len__(self):
#         return len(self.images)
#
#
#
# train_data  = trainset()

temp_sum = 0
cnt = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
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
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    residual = (X - dataset_global_mean) ** 2
    channel_var_mean = torch.mean(residual, dim=(0,2,3))
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += math.sqrt(channel_var_mean[0].item())
dataset_global_std = temp_sum / cnt
print('整个数据集的像素标准差:{}'.format(dataset_global_std))