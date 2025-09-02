import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import json
import gc
import numpy as np


def process_data_in_chunks(file_path, chunk_size = 10000, batch_size = 32):
    """
    将大文件分块处理，每个batch单独输出为一个JSON块
    :param file_path: 输入文件路径
    :param chunk_size: 每个数据块的大小
    :param batch_size: 批次大小
    """
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_file = os.path.join(base_dir, 'criteo/input_sparse.json')

    # 定义稀疏特征
    sparse_features = [f'C{i}' for i in range(1, 27)]

    # 使用上下文管理器写文件
    with open(output_file, 'w') as f:
        batch_data = []

        # 按块读取文件
        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, delimiter='\t', chunksize = chunk_size)):
            print(f"Processing chunk {chunk_idx + 1}")

            # 提取稀疏特征
            sparse_data = chunk[sparse_features]

            # 缺失值填充
            imputer = SimpleImputer(strategy = 'constant', fill_value = 'missing')
            sparse_data_filled = pd.DataFrame(imputer.fit_transform(sparse_data), columns = sparse_features)

            # 稀疏特征编码
            encoder = LabelEncoder()
            for feature in sparse_features:
                sparse_data_filled[feature] = encoder.fit_transform(sparse_data_filled[feature])

            # 转换为PyTorch Tensor
            sparse_tensor = torch.tensor(sparse_data_filled.values, dtype = torch.long)

            # 累积数据进入batch
            for i in range(0, len(sparse_tensor), batch_size):
                batch = sparse_tensor[i:i + batch_size]

                # 如果batch不满，进行填充
                if batch.shape[0] < batch_size:
                    padding = torch.zeros(batch_size - batch.shape[0], batch.shape[1], dtype = torch.long)
                    batch = torch.cat([batch, padding], dim = 0)

                # 构建JSON数据
                input_dict = {}
                for j in range(batch.shape[1]):
                    input_dict[f"input{j}"] = {
                        "shape": [batch_size, 1],
                        "data": batch[:, j].tolist()
                    }

                # 修改文件写入部分，使用json.dump进行格式化
                json.dump(input_dict, f, indent = 4)
                f.write('\n')

            # 清理内存
            del sparse_data, sparse_data_filled, sparse_tensor
            gc.collect()

    print(f"数据已处理完成，每个batch单独保存到 {output_file}")


# 使用示例
base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, 'criteo', 'criteo.txt')
process_data_in_chunks(file_path, batch_size = 32)