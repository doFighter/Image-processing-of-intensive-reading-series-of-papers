#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022/8/18 9:15
# @Author : doFighter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import time
from matplotlib import pyplot as plt


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, input):
        x = self.conv1(input)
        x = self.maxpool1(x)
        x = self.sigmoid(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.sigmoid(x)
        x = x.view(-1, 16*5*5)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        output = self.logSoftmax(x)
        return output
    

def train_runner(model, device, trainloader,crossLoss, optimizer, epoch):
    #训练模型, 将模型切换为训练模式，启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
    model.train()
    total = 0
    correct =0.0
    Loss = []
    Accuracy = []


    #enumerate迭代已加载的数据集,同时获取数据和数据下标
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        #把模型部署到device上
        inputs, labels = inputs.to(device), labels.to(device)
        #初始化梯度
        optimizer.zero_grad()
        #保存训练结果
        outputs = model(inputs)
        #计算损失和
        #多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题, 通常使用sigmod
        loss = crossLoss(outputs, labels)
        #获取最大概率的预测结果
        #dim=1表示返回每一行的最大值对应的列下标
        predict = outputs.argmax(dim=1)
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
        transforms.Resize((32,32)),
        #将图片转化为Tensor格式
        transforms.ToTensor(),
        #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)，正则化参数与数据库匹配，可以到官网查询
        transforms.Normalize((0.1307,),(0.3081,))    
    ])
    pipline_test = transforms.Compose([
        #将图片尺寸resize到32x32
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    #下载数据集
    train_set = datasets.MNIST(root="./datasets", train=True, download=True, transform=pipline_train)
    test_set = datasets.MNIST(root="./datasets", train=False, download=True, transform=pipline_test)
    #加载数据集
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    #创建模型，并获取设备可用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    #定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 定义损失函数
    crossLoss = nn.CrossEntropyLoss()
    # 调用模型进行训练并测试
    epoch = 5
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
    # titles[0:9]
    show_fashion_mnist(show_image, titles[0:9], pipline_test)

