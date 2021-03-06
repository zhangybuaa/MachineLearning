import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import sys
sys.path.append("/home/BH/zy1906304/zy/deeplearning/")
import d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/home/BH/zy1906304/zy/deeplearning/fine-tuning'

# train_imgs = ImageFolder(os.path.join(data_dir,'hotdog/train'))
# test_imgs = ImageFolder(os.path.join(data_dir,'hotdog/test'))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

# pretrained_net = models.resnet18(pretrained=False)
# pretrained_net.load_state_dict(torch.load(os.path.join(data_dir,'resnet18-5c106cde.pth')))
#
# pretrained_net.fc = nn.Linear(512, 2)
#
# output_params = list(map(id, pretrained_net.fc.parameters()))
# feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
#
# lr = 0.01
# optimizer = optim.SGD([{'params': feature_params},
#                        {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
#                        lr=lr, weight_decay=0.001)


scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)

def train_fine_tuning(net, optimizer, batch_size=256, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

# train_fine_tuning(pretrained_net, optimizer)
train_fine_tuning(scratch_net, optimizer)