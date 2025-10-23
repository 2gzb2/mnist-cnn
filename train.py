import os
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CNN import DigitCNN

# data：下载数据集
train_data = dataset.MNIST(root="mnist",
                           train=True,  # 作为训练集
                           transform=transforms.ToTensor(), # 转为tensor
                           download=True)

test_data = dataset.MNIST(root="mnist",
                           train=False,
                           transform=transforms.ToTensor(), # 转为tensor
                           download=False)

# batchsize:限制批次处理的数量
#创建数据加载器
train_loader = DataLoader(dataset=train_data,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(dataset=test_data,
                       batch_size=64,
                       shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = DigitCNN().to(device)

# loss:损失函数

loss_func = torch.nn.CrossEntropyLoss()

# optimizer:优化器

optimizer = torch.optim.Adam(cnn.parameters(),lr=0.01)

# training:训练
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = cnn(images)
        # 计算损失
        loss = loss_func(outputs,labels)

        optimizer.zero_grad()   # 梯度清零
        loss.backward()     # 反向传播
        optimizer.step()

    print("epoch is {}, "
          "ite is {}/{},"
          " loss is {}".format(epoch+1,i,
                               len(train_data)//64,
                               loss.item()))

# eval/test:测试
    loss_test = 0
    accuracy = 0
    cnn.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()

            outputs = cnn(images)
            loss_test += loss_func(outputs, labels)
            _, pred = outputs.max(1)
            accuracy += (pred == labels).sum().item()

        accuracy = accuracy / len(test_data)
        loss_test = loss_test / (len(test_data)//64)

        print("epoch is {}, accuracy is {}, "
              "loss_test is {}".format(epoch + 1,
                                       accuracy,
                                       loss_test.item()))

# save
os.makedirs("model", exist_ok=True)
torch.save(cnn, "model/mnist_model.pkl")