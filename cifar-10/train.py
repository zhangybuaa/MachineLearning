import torch
import torch.nn as nn
import torch.optim as optim
import model
import dataload
import argparse
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./checkpoint', help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCH = 2  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
LR = 0.1  # 学习率

# 模型定义-ResNet
net = model.ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
# 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 训练
if __name__ == "__main__":
    print("Start Training, Resnet-18!")
    num_iters = 0
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0
        for i, data in enumerate(dataload.trainloader, 0):
            # 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
            # 下标起始位置为0，返回 enumerate(枚举) 对象。

            num_iters += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空梯度

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)  # 选出每一列中最大的值作为预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 每20个batch打印一次loss和准确率
            if (i + 1) % 20 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, num_iters, sum_loss / (i + 1), 100. * correct / total))

        # print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in dataload.testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total
            # 将每次测试结果实时写入acc.txt文件中
            print('Saving model......')
            torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
            # f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
            # f.write('\n')
            # f.flush()
            # # 记录最佳测试分类准确率并写入best_acc.txt文件中
            # if acc > best_acc:
            #     f3 = open("best_acc.txt", "w")
            #     f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
            #     f3.close()
            #     best_acc = acc


    print("Training Finished, TotalEPOCH=%d" % EPOCH)