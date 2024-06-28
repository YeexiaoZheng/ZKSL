import onnx
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='Path to the ONNX model')
parser.add_argument('-o', '--output', type=str, default="", help='Path to the output json file')
args = parser.parse_args()
model_path = args.model
save_path = args.output if args.output else '../jsons/' + model_path.replace('.onnx', '.json')

model = onnx.load(model_path)
model_dict = {'tensor_map': {}, 'nodes': []}

# 提取模型的结构信息
# Constuct tensor map
for tensor in model.graph.initializer:
    tensor_name = tensor.name
    tensor = onnx.numpy_helper.to_array(tensor).T
    tensor_shape = list(tensor.shape) if tensor.ndim != 1 else [tensor.shape[0], 1]
    tensor = tensor.flatten().tolist()
    model_dict['tensor_map'][tensor_name] = {'shape': tensor_shape, 'data': tensor}

# Constuct input/output shape
model_dict['input_shape'] = [[dim.dim_value for dim in input.type.tensor_type.shape.dim] for input in model.graph.input][0]
model_dict['output_shape'] = [[dim.dim_value for dim in output.type.tensor_type.shape.dim] for output in model.graph.output][0]
model_dict['tensor_map']['input'] = {'shape': model_dict['input_shape'], 'data': np.random.rand(10).tolist()}

# Constuct node list
for node in model.graph.node:
    op_type = node.op_type
    attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    inputs = [input for input in node.input]
    outputs = [output for output in node.output]
    attributes = {attr.name: (attr.f, attr.type) for attr in node.attribute}
    attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    model_dict["nodes"].append({
        'op_type': op_type,
        'inputs': inputs,
        'outputs': outputs,
        'attributes': attributes
    })

print(model_dict)
json.dump(model_dict, open(save_path, 'w'))