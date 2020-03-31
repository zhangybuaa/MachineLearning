import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



import os
import pandas as pd
from PIL import Image
from data_augmentation import FixedRotation
from autoaugment import ImageNetPolicy
import numpy as np
from ranger import Ranger
import models


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '/home/BH/zy1906304/zy/deeplearning/foods'
resume = False


#数据增强
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.RandomRotation(15),
    transforms.RandomChoice([transforms.Resize([224, 224]), transforms.CenterCrop([224, 224])]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(),
    ImageNetPolicy(),
    # transforms.RandomLighting(0.1),
    transforms.ToTensor(),
    normalize
    ])

val_augs = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])


def default_loader(path,filename):
    # img_pil =  Image.open(path)
    # img_tensor = train_augs(img_pil)
    final_path = os.path.join(path, str(filename) + ".jpg")
    return Image.open(final_path).convert('RGB')
    # return img_tensor
#读入数据
class Trainset(Dataset):
    def __init__(self, label_list, transform=None, loader=default_loader):
        # 定义好 image 的路径
        images = pd.Series(label_list['filename']).values
        # images = [str(i)+".jpg" for i in images]
        # images = [os.path.join(data_dir+'train', i )for i in images]
        label = pd.Series(label_list['label']).values
        self.images = images
        self.label = label
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.images[index]
        img = self.loader('./data/train',filename)
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

class Valset(Dataset):
    def __init__(self, label_list, transform=None, loader=default_loader):
        # 定义好 image 的路径
        images = pd.Series(label_list['filename']).values
        # images = [str(i)+".jpg" for i in images]
        # images = [os.path.join(data_dir+'train', i )for i in images]
        label = pd.Series(label_list['label']).values
        self.images = images
        self.label = label
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.images[index]
        img = self.loader('./data/train' ,filename)
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(train_iter, test_iter, net,  feature_params, loss,device, num_epochs,file_name):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    best_test_acc = 0
    lr = 0.001
    optimizer = Ranger([{'params': feature_params},
                        {'params': net.fc.parameters(), 'lr': lr * 10}],
                             lr=lr, weight_decay=0.0001)
    # optimizer = optim.SGD([{'params': feature_params},
    #                        {'params': net.fc.parameters(), 'lr': lr * 10}],
    #                       lr=lr, weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLr(optimizer, T_max=5, eta_min=4e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(1, num_epochs+1):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        scheduler.step()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.5f, train_acc %.5f, val_acc %.5f, time %.1f sec'
              % (epoch , train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if test_acc > best_test_acc:
            print('find best! save at model/%s/best.pth'%file_name)
            best_test_acc = test_acc
            torch.save(net.state_dict(), './model/%s/best.pth'%file_name)
            with open('./result/%s.txt' % file_name, 'a') as acc_file:
                acc_file.write('Epoch: %2d, acc: %.8f\n' % (epoch, test_acc))
        if epoch%10==0:
            torch.save(net.state_dict(), './model/%s/checkpoint_%d.pth' % (file_name,epoch))



# pretrained_net= torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet50_2', pretrained=True)

pretrained_net = models. wide_resnet50_2(pretrained=True)
# pretrained_net.load_state_dict(torch.load(os.path.join(data_dir,'resnet50-19c8e357.pth')))

pretrained_net.fc = nn.Linear(2048, 4)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
#
num_epochs, batch_size, weight_decay = 40, 32, 0.001
loss = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD([{'params': feature_params},
#                        {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
#                        lr=lr, weight_decay=0.001)

# def train_fine_tuning(net, optimizer, batch_size, num_epochs,train_data,val_data):
#     train_iter = DataLoader(train_data,batch_size, shuffle=True)
#     test_iter = DataLoader(val_data,batch_size,shuffle=False)
#     loss = torch.nn.CrossEntropyLoss()
#     train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

for index in range (1,6):
    print ('fold: %d' %index)
    file_name = "resnet18_{}".format(index)
    if not os.path.exists('./model/%s' % file_name):
        os.makedirs('./model/%s' % file_name)
    # if not os.path.exists('./result/%s' % file_name):
    #     os.makedirs('./result/%s' % file_name)

    if not os.path.exists('./result/%s.txt' % file_name):
        txt_mode = 'w'
    else:
        txt_mode = 'a'
    with open('./result/%s.txt' % file_name, txt_mode) as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))

    train_data_list = pd.read_csv("./data/k_folds/train_{}.csv".format(index), sep=",")
    val_data_list = pd.read_csv("./data/k_folds/test_{}.csv".format(index), sep=",")
    train_data = Trainset(train_data_list,transform=train_augs)
    val_data = Valset(val_data_list,transform=val_augs)
    train_iter = DataLoader(train_data, batch_size, shuffle=True)
    test_iter = DataLoader(val_data, batch_size, shuffle=False)
    if resume:
        net = models.resnet50(pretrained=False)
        net.fc = nn.Linear(2048, 4)
        net.load_state_dict(torch.load('./model/resnet18_5/best.pth'))
        train(train_iter, test_iter, net, feature_params, loss, device, num_epochs, file_name)
    else:
        train(train_iter, test_iter, pretrained_net,feature_params, loss, device, num_epochs,file_name)


    # train_fine_tuning(pretrained_net, optimizer,100,20,train_data,val_data )
    # train_fine_tuning(scratch_net, optimizer)