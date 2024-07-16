import torch
from torchvision import datasets, transforms
import json

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 取出前100个样本
num_samples = 10
input_size = 28*28
cut_input_size = 200
start = int((input_size - cut_input_size) / 2)
data = []

for i in range(num_samples):
    image, label = mnist_dataset[i]
    image = image.view(-1).numpy().tolist()[start:start + cut_input_size]  # 将图像展平并转换为列表
    data.append({"data": image, "label": label})

# 保存为JSON文件
with open('input.json', 'w') as f:
    json.dump(data, f)

print("input.json has been created with 100 MNIST samples")
