import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import json
import gc


def analyze_features(file_path, chunk_size=10000):
    """
    Analyze the basic statistical information of all features.
    :param file_path: Input file path.
    :param chunk_size: Size of each data chunk.
    :return: Dictionary of feature statistical information.
    """
    feature_stats = {}

    # Read the first chunk to get column names
    first_chunk = pd.read_csv(file_path, delimiter='\t', nrows=1)
    all_features = [col for col in first_chunk.columns if col != 'Status']

    # Initialize feature statistical information
    for feature in all_features:
        feature_stats[feature] = {'unique_values': set(), 'count': 0}

    # Read and count in chunks
    for chunk in pd.read_csv(file_path, delimiter='\t', chunksize=chunk_size):
        for feature in all_features:
            # Convert feature values to strings for unified processing
            unique_vals = set(chunk[feature].astype(str).dropna().unique())
            feature_stats[feature]['unique_values'].update(unique_vals)
            feature_stats[feature]['count'] = len(feature_stats[feature]['unique_values'])

    # Convert set to count
    for feature in feature_stats:
        print(f"Feature {feature} has {feature_stats[feature]['count']} unique values")

    return feature_stats


def process_data_in_chunks(file_path, chunk_size=10000, batch_size=32):
    """
    Process all features and save them as batch files.
    :param file_path: Input file path.
    :param chunk_size: Size of each data chunk.
    :param batch_size: Batch size.
    """
    # Get the root directory of the project
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, 'loan_default', 'batch_files')

    # Create the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Analyze features first
    print("Starting to analyze feature statistical information...")
    feature_stats = analyze_features(file_path, chunk_size)

    # Save feature statistical information
    stats_file = os.path.join(base_dir, 'loan_default', 'feature_stats.json')
    with open(stats_file, 'w') as f:
        # Convert set to list for JSON serialization
        serializable_stats = {
            k: {
                'count': v['count']
            } for k, v in feature_stats.items()
        }
        json.dump(serializable_stats, f, indent=4)

    # Process data in chunks
    for chunk_idx, chunk in enumerate(pd.read_csv(file_path, delimiter='\t', chunksize=chunk_size)):
        print(f"Processing chunk {chunk_idx + 1}")

        # Separate features and labels
        features = chunk.drop('Status', axis=1)
        labels = chunk['Status']

        # Fill missing values
        imputer = SimpleImputer(strategy='constant', fill_value='missing')
        features_filled = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

        # Convert all features to strings for unified processing
        for col in features_filled.columns:
            features_filled[col] = features_filled[col].astype(str)

        # Feature encoding
        encoded_features = pd.DataFrame()
        encoders = {}
        for feature in features_filled.columns:
            encoder = LabelEncoder()
            encoded_features[feature] = encoder.fit_transform(features_filled[feature])
            encoders[feature] = encoder

        # Convert to PyTorch Tensor
        features_tensor = torch.tensor(encoded_features.values, dtype=torch.long)
        labels_tensor = torch.tensor(labels.values, dtype=torch.long)

        # Process and save by batch
        for batch_idx in range(0, len(features_tensor), batch_size):
            batch = features_tensor[batch_idx:batch_idx + batch_size]
            label_batch = labels_tensor[batch_idx:batch_idx + batch_size]

            # Build batch dictionary
            batch_dict = {
                "input": {
                    "shape": list(batch.shape),
                    "data": batch.reshape(-1).tolist()
                },
                "label": {
                    "shape": [batch.shape[0], 1],
                    "data": label_batch.tolist()
                },
                "inputs": [
                    {
                        "shape": [batch.shape[0], 1],
                        "data": batch[:, j].tolist()
                    } for j in range(batch.shape[1])
                ],
                "feature_names": list(features.columns)  # Add feature name information
            }

            # Save batch
            batch_file = os.path.join(output_dir,
                                      f"batch_{chunk_idx * (len(features_tensor) // batch_size) + batch_idx // batch_size}.json")
            try:
                with open(batch_file, 'w') as f:
                    json.dump(batch_dict, f, indent=4)
            except Exception as e:
                print(f"Error occurred when saving batch {batch_idx}: {str(e)}")

        # Clean up memory
        del features, features_filled, features_tensor, labels_tensor
        gc.collect()

    print(f"Data processing completed. All batch files have been saved to {output_dir}")
    print(f"Feature statistical information has been saved to {stats_file}")


# Example usage
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, 'loan_default', 'loan_default_small.txt')
    process_data_in_chunks(file_path, batch_size=32)