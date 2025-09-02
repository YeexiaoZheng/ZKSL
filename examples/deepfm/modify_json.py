import json

# Load the input JSON
with open('model_1129.json', 'r') as file:
    data = json.load(file)

# Ensure data contains 'nodes' key
if 'nodes' not in data or not isinstance(data['nodes'], list):
    raise TypeError("The input JSON must contain a 'nodes' key with a list of dictionaries.")

nodes = data['nodes']
new_nodes = []


def merge_gather_nodes(node1, node2):
    merged_inputs = [inp for inp in (node1.get('inputs', []) + node2.get('inputs', [])) if
                     'input' in inp or 'embedding' in inp]
    return {
        'op_type': 'Gather',
        'inputs': merged_inputs,
        'outputs': node2.get('outputs', []),
        'attributes': node1.get('attributes', {}),
        'backward_inputs': [],
        'backward_outputs': [],
    }


def process_unsqueeze_node(node):
    new_inputs = [inp for inp in node.get('inputs', []) if 'Constant' not in inp]
    return {
        'op_type': node['op_type'],
        'inputs': new_inputs,
        'outputs': node.get('outputs', []),
        'attributes': node.get('attributes', {}),
        'backward_inputs': [],
        'backward_outputs': [],
    }


skip_next = False
for idx in range(len(nodes) - 1):
    if skip_next:
        skip_next = False
        continue

    current_node = nodes[idx]
    next_node = nodes[idx + 1]

    if current_node.get('op_type') == 'Gather' and next_node.get('op_type') == 'Gather':
        merged_node = merge_gather_nodes(current_node, next_node)
        new_nodes.append(merged_node)
        skip_next = True
    else:
        if current_node.get('op_type') == 'Unsqueeze':
            processed_node = process_unsqueeze_node(current_node)
            new_nodes.append(processed_node)
        else:
            new_nodes.append(current_node)

# Add the last node if it was not merged
if not skip_next and len(nodes) > 0:
    last_node = nodes[-1]
    if last_node.get('op_type') == 'Unsqueeze':
        new_nodes.append(process_unsqueeze_node(last_node))
    else:
        new_nodes.append(last_node)

# Update the data with the new nodes
data['nodes'] = new_nodes

# Write the new JSON to an output file
with open('output.json', 'w') as file:
    json.dump(data, file, indent=2)

print("Processed nodes successfully written to output.json")

