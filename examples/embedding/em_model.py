import torch.nn as nn
import os
import json
import torch
from glob import glob


class EM_Model(nn.Module):
    def __init__(self, sparse_feature_number, embedding_dim):
        super(EM_Model, self).__init__()
        # Define the shared Embedding layer
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=feature_size, embedding_dim=embedding_dim)
            for feature_size in sparse_feature_number
        ])

    def forward(self, sparse_inputs):
        # The Embedding layer is shared to calculate the embedding vector of each feature
        embeddings = torch.stack([
            embedding(sparse_inputs[:, i])
            for i, embedding in enumerate(self.embedding_layers)
        ], dim=1)  # shape: (batch_size, num_fields, embedding_dim)
        return embeddings


def load_feature_stats(base_dir):
    """
    Load feature statistical information
    """
    stats_file = os.path.join(base_dir, 'criteo', 'feature_stats.json')
    with open(stats_file, 'r') as f:
        feature_stats = json.load(f)
    return feature_stats


def get_batch_files(base_dir):
    """
    Get the paths of all batch files
    """
    batch_path = os.path.join(base_dir, 'criteo', 'batch_files')
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


def main():
    # Get the root directory of the project
    base_dir = os.path.dirname(os.path.dirname(__file__))

    # Load feature statistical information
    feature_stats = load_feature_stats(base_dir)

    # Get the list of the number of features (the number of unique values for each feature)
    sparse_feature_number = [stats['count'] + 1 for stats in feature_stats.values()]  # +1 for missing values

    # Hyperparameters
    embedding_dim = 9  # Embedding dimension

    # Initialize the model
    model = EM_Model(
        sparse_feature_number=sparse_feature_number,
        embedding_dim=embedding_dim,
    )

    print("Model architecture:")
    print(model)

    # Get all batch files
    batch_files = get_batch_files(base_dir)

    # Read data for ONNX export
    if batch_files:
        all_input_data = []
        all_label_data = []

        # Load the data of all batches
        for batch_file in batch_files:
            input_data, label_data = load_batch(batch_file)
            all_input_data.append(input_data)
            all_label_data.append(label_data)

        # Combine all input data
        combined_input_data = torch.cat(all_input_data, dim=0)

        # Export the model to ONNX
        torch.onnx.export(
            model,
            combined_input_data,
            "model.onnx",
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("Model has been exported to model.onnx")
    else:
        print("No batch files found!")


if __name__ == "__main__":
    main()