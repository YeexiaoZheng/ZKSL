import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        # x = self.fc1(x)
        x = x.mean(dim=1)  # 对嵌入进行平均
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Generate random training data
vocab_size = 10
embedding_dim = 3
hidden_dim = 32
output_dim = 2
batch_size = 32
sequence_length = 10
num_samples = 1000

# Random input data (token indices) and random target labels
inputs = torch.randint(0, vocab_size, (num_samples, 10))  # 每个句子长度为10
labels = torch.randint(0, 2, (num_samples,))  # 二分类标签

# Create a DataLoader
dataset = torch.utils.data.TensorDataset(inputs, labels)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, criterion, and optimizer
model = EmbeddingModel(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Export the model to ONNX
dummy_input = torch.randint(0, vocab_size, (1, sequence_length))
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("Model has been exported to model.onnx")
