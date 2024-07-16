use std::collections::HashMap;

use ndarray::IxDyn;

use crate::utils::{
    helpers::Tensor,
    loader::{GraphJson, Input, NodeJson},
    math::Int,
};

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub backward_inputs: Vec<String>,
    pub backward_outputs: Vec<String>,
    pub attributes: HashMap<String, f64>,
}

impl Node {
    pub fn construct(node_json: NodeJson) -> Self {
        Self {
            op_type: node_json.op_type,
            inputs: node_json.inputs,
            outputs: node_json.outputs,
            backward_inputs: node_json.backward_inputs,
            backward_outputs: node_json.backward_outputs,
            attributes: node_json.attributes.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub tensor_map: HashMap<String, Tensor>,
    pub nodes: Vec<Node>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl Graph {
    pub fn construct(graph_json: GraphJson, scale_factor: u64) -> Self {
        Self {
            tensor_map: graph_json
                .tensor_map
                .into_iter()
                .map(|(s, tensor)| {
                    (
                        s.clone(),
                        Tensor::from_shape_vec(
                            IxDyn(&tensor.shape),
                            tensor
                                .data
                                .iter()
                                .map(|x| (x * scale_factor as f64) as Int)
                                .collect(),
                        )
                        .unwrap(),
                    )
                })
                .collect(),
            nodes: graph_json.nodes.into_iter().map(Node::construct).collect(),
            input_shape: graph_json.input_shape,
            output_shape: graph_json.output_shape,
        }
    }

    pub fn run(&self) -> Tensor {
        for _node in self.nodes.iter() {
            todo!();
        }
        Tensor::from_shape_vec(IxDyn(&[1, 2]), vec![1, 2]).unwrap()
    }
}

pub struct GraphInput {
    pub data: Tensor,
    pub label: Int,
}

impl GraphInput {
    pub fn construct(inputs: Vec<Input>, scale_factor: u64) -> Vec<Self> {
        inputs
            .iter()
            .map(|input| Self {
                data: Tensor::from_shape_vec(
                    IxDyn(&[1, input.data.len()]),
                    input
                        .data
                        .iter()
                        .map(|x| ((x * scale_factor as f64) as Int))
                        .collect(),
                )
                .unwrap(),
                label: input.label as Int,
            })
            .collect()
    }
}
