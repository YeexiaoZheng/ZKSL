use std::collections::HashMap;

use ndarray::IxDyn;

use crate::layers::layer::Tensor;
use crate::utils::loader::{GraphJson, NodeJson};

pub struct Node {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, String>,
}

impl Node {
    pub fn construct(node_json: NodeJson) -> Self {
        Self {
            op_type: node_json.op_type.clone(),
            inputs: node_json.inputs.clone(),
            outputs: node_json.outputs.clone(),
            attributes: node_json.attributes.clone(),
        }
    }
}

pub struct Graph {
    pub tensor_map: HashMap<String, Tensor>,
    pub nodes: Vec<Node>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl Graph {
    pub fn construct(graph_json: GraphJson) -> Self {
        Self {
            tensor_map: graph_json
                .tensor_map
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        Tensor::from_shape_vec(IxDyn(&v.shape), v.data.clone()).unwrap(),
                    )
                })
                .collect(),
            nodes: graph_json
                .nodes
                .iter()
                .map(|node_json| Node::construct(node_json.clone()))
                .collect(),
            input_shape: graph_json.input_shape,
            output_shape: graph_json.output_shape,
        }
    }

    pub fn run(&self) -> Tensor {
        for node in self.nodes.iter() {
            todo!();
        }
        Tensor::from_shape_vec(IxDyn(&[1, 2]), vec![1, 2]).unwrap()
    }
}
