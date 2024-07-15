import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 生成随机输入数据
num_samples = 1000
input_dim = 10

# 输入数据，每个样本是一个大小为10的一维向量
X_train = np.random.rand(num_samples, input_dim).astype(np.float32)

# 生成随机二分类标签（0或1）
y_train = np.random.randint(2, size=num_samples).astype(np.int64)

# PyTorch 模型定义
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.mlp1 = nn.Linear(input_dim, 10)
        self.mlp2 = nn.Linear(10, 2)
    
    def forward(self, x):
        x1 = torch.relu(self.mlp1(x))
        x2 = self.mlp2(x1)
        output = torch.softmax(x2, dim=1)
        return output

# 创建模型实例
model = SimpleModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据加载器
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    outputs = model(torch.from_numpy(X_train))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.from_numpy(y_train)).sum().item() / num_samples
    print(f'训练集上的准确率: {accuracy:.4f}')

# 转换为 ONNX 格式
onnx_model_file = 'model.onnx'
dummy_input = torch.randn(1, input_dim)
torch.onnx.export(model, dummy_input, onnx_model_file, input_names=['input'], output_names=['output'])

print(f'ONNX 模型已保存到 {onnx_model_file}')