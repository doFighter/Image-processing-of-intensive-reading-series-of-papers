#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2023/1/20 13:43
# @Author : doFighter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import time
from matplotlib import pyplot as plt

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **args):
        super(Inception, self).__init__(**args)
        self.conv1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.conv2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.conv2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.conv3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.pool4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.relu(self.conv1_1(x))
        conv2 = self.relu(self.conv2_2(self.relu(self.conv2_1(x))))
        conv3 = self.relu(self.conv3_2(self.relu(self.conv3_1(x))))
        conv4 = self.relu(self.conv4_2(self.pool4_1(x)))
        return torch.cat((conv1, conv2, conv3, conv4), dim=1)
    
# 定义辅组分类器
class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        # 原模型
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
 
    def forward(self, x):
        x = self.averagePool(x)
        x = self.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class GoogLeNet(nn.Module):
	# 传入的参数中aux_logits=True表示训练过程用到辅助分类器，aux_logits=False表示验证过程不用辅助分类器
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.feature1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Inception输出时已经使用relu激活，因此后面不再需要使用
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(3, stride=2, padding=1),
            Inception(480, 192, (96, 208), (16, 48), 64)
        )
        # self.feature2 = nn.Sequential(
        #     Inception(512, 160, (112, 224), (24, 64), 64),
        #     Inception(512, 128, (128, 256), (24, 64), 64),
        #     Inception(512, 112, (144, 288), (32, 64), 64)
        # )
        # self.feature3 = nn.Sequential(
        #     Inception(528, 256, (160, 320), (32, 128), 128),
        #     nn.MaxPool2d(3, stride=2, padding=1),
        #     Inception(832, 256, (160, 320), (32, 128), 128),
        #     Inception(832, 384, (192, 384), (48, 128), 128)
        # )
 
        # if self.aux_logits:
        #     self.aux1 = AuxClassifier(512, num_classes)
        #     self.aux2 = AuxClassifier(528, num_classes)

        self.classifier = nn.Sequential(
            nn.AvgPool2d(2, stride=1),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
 
    def forward(self, x):
        x = self.feature1(x)
        # if self.training and self.aux_logits:    # eval model lose this layer
        #     aux1 = self.aux1(x)
        # x = self.feature2(x)
        # if self.training and self.aux_logits:    # eval model lose this layer
        #     aux2 = self.aux2(x)
        # x = self.feature3(x)
        x = self.classifier(x)
        # if self.training and self.aux_logits:   # eval model lose this layer
        #     return x, aux1, aux2
        return x
    
def train_runner(model, device, trainloader,crossLoss, optimizer, epoch):
    #训练模型, 将模型切换为训练模式，启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
    model.train()
    total = 0
    correct = 0.0
    Loss = []
    Accuracy = []


    #enumerate迭代已加载的数据集,同时获取数据和数据下标
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        #把模型部署到device上
        inputs, labels = inputs.to(device), labels.to(device)
        #初始化梯度
        optimizer.zero_grad()
        #训练时有两个辅助分类器，因此一共输出3个结果，保存训练结果
        # logits, aux_logits1, aux_logits2 = model(inputs)
        # #计算损失和
        # #多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题, 通常使用sigmod
        # loss1 = crossLoss(logits, labels)
        # loss2 = crossLoss(aux_logits1, labels)
        # loss3 = crossLoss(aux_logits2, labels)
        # # 分别计算各分类器的损失，最后计算总损失，辅助分类器损失权重因子为0.3
        # loss = loss1 + loss2 * 0.3 + loss3 * 0.3
        #获取最大概率的预测结果，这里只获取主干分类器的结果

        output = model(inputs)
        loss = crossLoss(output, labels)
        predict = output.argmax(dim=1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()
        if i % 1000 == 0:
            #loss.item()表示当前loss的数值
            print("Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, loss.item(), 100*(correct/total)))
            Loss.append(loss.item())
            Accuracy.append(correct/total)
    return loss.item(), correct/total

def test_runner(model, device, testloader, crossLoss):
    #模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
    #因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model.eval()
    #统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0
    #torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += crossLoss(output, label).item()
            predict = output.argmax(dim=1)
            #计算正确数量
            total += label.size(0)
            correct += (predict == label).sum().item()
        #计算损失值
        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/total, 100*(correct/total)))
    
    return test_loss/total, correct/total


def transform_convert(img_tensor, transform):
    """
    该函数将加载的图像数据转为普通图像，即逆操作
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    # 判断图像是否经过标准化操作，若是，则进行逆标准化操作还原图像
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    # 如果数据被加载在GPU内，将数据切换到CPU
    if img_tensor.device.type == "cuda":
        img_tensor = img_tensor.cpu()

    # 将tensor类型的图像数据的进行形状上的转换(通道x行x列---》行x列x通道)
    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    # 当图像执行了归一化操作时，对图像像素进行还原
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    # 当图像数据结构类型是tensor时，将其转为numpy类型
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    # 判断图像通道数目，并转换成对应类型图像
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        # 单通道输入二维数据，删除通道项，否则会报错
        img = Image.fromarray(img_tensor[:, :, 0].astype('uint8'))
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img

def show_fashion_mnist(images, labels, transform):
#     labels = get_fashion_mnist_labels(labels)
    _,figs = plt.subplots(1, len(images), figsize=(6, 6))
    for f, img, lbs in zip(figs, images, labels):
        img = transform_convert(img, transform)
        f.imshow(img)
        f.set_title(lbs)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()



if __name__ == "__main__":
    pipline_train = transforms.Compose([
        #随机旋转图片
        transforms.RandomHorizontalFlip(),
        #将图片尺寸resize到32x32
        # transforms.Resize((32,32)),
        #将图片转化为Tensor格式
        transforms.ToTensor(),
        #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)，正则化参数与数据库匹配，可以到官网查询
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))    
    ])
    pipline_test = transforms.Compose([
        #将图片尺寸resize到32x32
        # transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    #下载数据集
    train_set = datasets.CIFAR100(root="./datasets", train=True, download=True, transform=pipline_train)
    test_set = datasets.CIFAR100(root="./datasets", train=False, download=True, transform=pipline_test)
    #加载数据集
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    #创建模型，并获取设备可用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet(num_classes=100).to(device)
    #定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    # 定义损失函数
    crossLoss = nn.CrossEntropyLoss()
    # 调用模型进行训练并测试
    epoch = 100
    TrainLoss = []
    TrainAccuracy = []
    TestLoss = []
    TestAccuracy = []
    for epoch in range(1, epoch+1):
        print("start_time",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        loss, acc = train_runner(model, device, trainloader, crossLoss, optimizer, epoch)
        TrainLoss.append(loss)
        TrainAccuracy.append(acc)
        loss, acc = test_runner(model, device, testloader, crossLoss)
        print("end_time: ",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'\n')
        TestLoss.append(loss)
        TestAccuracy.append(acc)

    # 训练结束，现实模型训练过程中的损失和准确率
    print('Finished Training')
    plt.subplot(1,2,1)
    plt.plot(TrainLoss, label="TrainLoss")
    plt.plot(TestLoss, label="TestLoss")
    plt.title('TrainLoss And TestLoss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(TrainAccuracy, label="TrainAccuracy")
    plt.plot(TestAccuracy, label="TestAccuracy")
    plt.title('TrainAccuracy And TestAccuracy')
    plt.legend()
    plt.show()

    # 通过读取部分图片，展示预测结果和真实label
    for x, y in testloader:
        break
    
    true_labels = y.detach().numpy()
    x = x.to(device)
    model.eval()
    pred_labels = model(x).argmax(dim=1)
    pred_labels = pred_labels.cpu().detach().numpy()

    titles = ['(T)' + str(true) + '\n' + '(P)' + str(pred) for true, pred in 
            zip(true_labels, pred_labels)]

    show_image = [x[i] for i in range(9)]
    show_fashion_mnist(show_image, titles[0:9], pipline_test)

 

