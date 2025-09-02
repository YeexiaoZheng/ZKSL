import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 卷积层1：输入1通道（灰度图），输出6通道，卷积核大小5x5，步幅1，填充2
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)

        # 池化层1：使用2x2的最大池化
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层2：输入6通道，输出16通道，卷积核大小5x5，步幅1，填充0
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 池化层2：使用2x2的最大池化
        self.pool2 = nn.MaxPool2d(2, 2)

        # 全连接层1：将卷积输出的16x5x5展平成一个向量，输出120个节点
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # 全连接层2：输入120个节点，输出84个节点
        self.fc2 = nn.Linear(120, 84)

        # 全连接层3：输入84个节点，输出10个类别（适用于MNIST等10分类问题）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积层1 + 池化层1 + 激活函数
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # 卷积层2 + 池化层2 + 激活函数
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # 展平层：将2D图像展平成1D
        x = x.view(-1, 16 * 5 * 5)

        # 全连接层1 + 激活函数
        x = F.relu(self.fc1(x))

        # 全连接层2 + 激活函数
        x = F.relu(self.fc2(x))

        # 全连接层3 + 输出
        x = self.fc3(x)

        return x

# 测试模型
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")



    transform = transforms.Compose([transforms.ToTensor(),])
    mnist_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    image, label = mnist_dataset[2]
    dummy_input =  image.unsqueeze(0)
    print(dummy_input.size())
    onnx_filename = "model_lenet.onnx"
    torch.onnx.export(model, dummy_input, onnx_filename, input_names=['input'], output_names=['output'])
    print(f'ONNX 模型已保存到 {onnx_filename}')
