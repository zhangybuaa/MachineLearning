import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import models
import os
from torch import nn
import pandas as pd
from PIL import Image
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
     # transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
    ])

def default_loader(root_dir,path):
    final_path = os.path.join(root_dir, str(path) + ".jpg")
    return Image.open(final_path).convert('RGB')


class Testset(Dataset):
    def __init__(self, transform=None,loader=default_loader):
        # 定义好 image 的路径
        images = []
        # label = []
        for i in range (856):
            images.append(i)
            # label.append(0)
        self.images = images
        # self.label = label
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader('./data/test',fn)
        # label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)



test_data  = Testset(transform=test_augs)

test_iter = DataLoader(test_data, batch_size=128, shuffle=False)

net = models.wide_resnet50_2(pretrained=False)
net.fc = nn.Linear(2048, 4)
net.load_state_dict(torch.load('./model/resnet18_1/best.pth'))
net = net.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

preds_list = []
id= 0
with torch.no_grad():
    for X in test_iter:
        batch_pred = list(net(X.to(device)).argmax(dim=1).cpu().numpy())
        for y_pred in batch_pred:
            preds_list.append((id, y_pred))
            id += 1

print('生成提交结果文件')
with open('submission.csv', 'w') as f:
    # f.write('ID,Prediction\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))
