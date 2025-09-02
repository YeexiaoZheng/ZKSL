import json
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import os


def load_feature_stats(base_dir):
    """
    Load feature statistical information
    """
    stats_file = os.path.join(base_dir, 'loan_default', 'feature_stats.json')
    with open(stats_file, 'r') as f:
        feature_stats = json.load(f)
    return feature_stats


def get_batch_files(base_dir):
    """
    Get the paths of all batch files
    """
    batch_path = os.path.join(base_dir, 'loan_default', 'batch_files')
    return sorted([f for f in glob(os.path.join(batch_path, 'batch_*.json'))])


def load_batch(batch_file):
    """
    Load the data of a single batch
    """
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)

    # Convert to tensor
    input_shape = batch_data['input']['shape']
    input_data = torch.tensor(batch_data['input']['data'], dtype=torch.long).reshape(input_shape)

    label_shape = batch_data['label']['shape']
    label_data = torch.tensor(batch_data['label']['data'], dtype=torch.long).reshape(label_shape)

    return input_data, label_data


class DeepFM(nn.Module):
    def __init__(self, sparse_feature_number, embedding_dim, hidden_dims, output_dim):
        super().__init__()
        # FM部分参数
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(feature_size, embedding_dim)
            for feature_size in sparse_feature_number
        ])

        # Deep部分参数
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], output_dim)
        )

        # fm和deep输出标准化，再引入可学习融合权重
        self.fm_bn = nn.BatchNorm1d(1)
        self.deep_bn = nn.BatchNorm1d(1)
        self.combine_weight = nn.Parameter(torch.ones(2))

        # 初始化
        for emb in self.embedding_layers:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, sparse_inputs):
        # FM部分（阻断Embedding梯度）
        with torch.no_grad():
            embeddings_nograd = torch.stack([
                self.embedding_layers[i](sparse_inputs[:, i])
                for i in range(len(self.embedding_layers))
            ], dim=1)

            # 一阶项
            fm_first_order = torch.sum(sparse_inputs.unsqueeze(-1).float(), dim=1)

            # 二阶项
            sum_emb = torch.sum(embeddings_nograd, dim=1)
            square_sum = torch.sum(embeddings_nograd ** 2, dim=1)
            fm_second_order = 0.5 * torch.sum(sum_emb ** 2 - square_sum, dim=1, keepdim=True)

            fm_output = fm_first_order + fm_second_order

        # Deep部分（正常梯度）
        embeddings = torch.stack([
            self.embedding_layers[i](sparse_inputs[:, i])
            for i in range(len(self.embedding_layers))
        ], dim=1)
        deep_output = self.mlp(torch.mean(embeddings, dim=1))

        fm_output = self.fm_bn(fm_output.unsqueeze(1)).squeeze()
        deep_output = self.deep_bn(deep_output.unsqueeze(1)).squeeze()

        combined = self.combine_weight[0] * fm_output + self.combine_weight[1] * deep_output
        # print("fm_output:", fm_output)
        # print("deep_output", deep_output)
        # print(combined.shape)

        return torch.sigmoid(combined).unsqueeze(-1)


def print_parameter_changes(model, initial_params, threshold=1e-6):
    """打印各层参数变化并标记显著变化的层"""
    print("\n" + "=" * 50)
    print("参数更新状态检查".center(50))
    print("=" * 50)

    max_name_length = max(len(name) for name in model.named_parameters())

    for name, param in model.named_parameters():
        initial = initial_params[name]
        current = param.data

        # 计算变化量
        delta = torch.norm(current - initial).item()  # L2范数
        max_change = torch.max(torch.abs(current - initial)).item()

        status = "✅ 已更新" if delta > threshold else "❌ 未更新"
        color_code = "\033[32m" if delta > threshold else "\033[31m"

        # 打印参数信息
        print(f"{color_code}▏ {name.ljust(max_name_length)} | "
              f"形状: {str(list(param.shape)).ljust(15)} | "
              f"变化量: {delta:.3e} | 最大变化: {max_change:.3e} | {status}\033[0m")

        # 打印前3个元素的变化（针对低维参数）
        if param.ndim <= 2 and param.numel() > 1:
            print(f"    Initial[:3]: {initial.flatten()[:3].detach().cpu().numpy().round(4)}")
            print(f"    Current[:3]: {current.flatten()[:3].detach().cpu().numpy().round(4)}")
            print("-" * 80)

def train(model, train_loader, val_loader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        print_parameter_changes(model, initial_params)

        # 验证阶段
        val_metrics = evaluate(model, val_loader, device)

        # 打印指标
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {total_loss / len(train_loader.dataset):.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        print("Confusion Matrix:")
        print(val_metrics["confusion_matrix"])
        print("-" * 50)

        # # 保存最佳模型
        # if val_metrics["f1"] > best_f1:
        #     best_f1 = val_metrics["f1"]
        #     torch.save(model.state_dict(), "best_model.pth")


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()

            preds = (outputs > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # print("pred:", all_preds)
    # print("labels:", all_labels)

    # 计算指标
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "confusion_matrix": cm,
        "specificity": tn / (tn + fp) if (tn + fp) != 0 else 0
    }


def main():
    # 数据加载部分保持不变...
    base_dir = os.path.dirname(os.path.dirname(__file__))
    feature_stats = load_feature_stats(base_dir)
    sparse_feature_number = [stats['count'] + 1 for stats in feature_stats.values()]

    # 正确初始化模型
    model = DeepFM(
        sparse_feature_number=sparse_feature_number,
        embedding_dim=9,
        hidden_dims=[64],
        output_dim=1
    )

    # 数据准备
    batch_files = get_batch_files(base_dir)
    if not batch_files:
        print("No batch files found!")
        return

    # 合并数据
    all_inputs, all_labels = [], []
    for batch_file in batch_files:
        inputs, labels = load_batch(batch_file)
        all_inputs.append(inputs)
        all_labels.append(labels)

    dataset = TensorDataset(
        torch.cat(all_inputs),
        torch.cat(all_labels).float()  # BCELoss需要float标签
    )

    # 数据分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 训练验证
    train(model, train_loader, val_loader, num_epochs=10, lr=0.00001)

    # 最终测试
    print("\nFinal Evaluation:")
    final_metrics = evaluate(model, val_loader, next(model.parameters()).device)
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision/Recall: {final_metrics['precision']:.4f}/{final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"Specificity: {final_metrics['specificity']:.4f}")
    print("Confusion Matrix:")
    print(final_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
