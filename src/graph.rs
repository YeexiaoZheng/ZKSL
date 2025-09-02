use std::collections::BTreeMap;

use log::debug;
use ndarray::IxDyn;

use crate::utils::{
    helpers::Tensor,
    loader::{GraphJson, NodeJson},
    math::Int,
};

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub backward_inputs: Vec<String>,
    pub backward_outputs: Vec<String>,
    pub attributes: BTreeMap<String, Vec<f64>>,
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
    pub tensor_map: BTreeMap<String, Tensor>,
    pub nodes: Vec<Node>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl Graph {
    pub fn construct(graph_json: GraphJson, scale_factor: u64, input_scale: bool) -> Self {
        Self {
            tensor_map: graph_json
                .tensor_map
                .into_iter()
                .map(|(s, tensor)| {
                    debug!("Parsing tensor: {:?}", s);
                    // info!("Parsing tensor: {:?}", s);
                    let scale = match input_scale {
                        true => scale_factor as f64,
                        false => {
                            if s.ends_with(".weight") || s.ends_with(".bias") {
                                scale_factor as f64
                            } else {
                                1.0
                            }
                        }
                    };

                    (
                        s.clone(),
                        Tensor::from_shape_vec(
                            IxDyn(&tensor.shape),
                            tensor.data.iter().map(|x| (x * scale) as Int).collect(),
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

    pub fn construct_sub_forward_graph(
        nodes: Vec<Node>,
        tensor_map: BTreeMap<String, Tensor>,
    ) -> (Self, Tensor) {
        let sub_inputs_str = nodes
            .iter()
            .map(|node| node.inputs.clone())
            .flatten()
            .collect::<Vec<_>>();
        let sub_inputs = match tensor_map.get(&nodes[0].inputs[0]) {
            Some(x) => x,
            None => panic!(
                "Error occurs at test_cnn: tensor '{}' not found",
                &nodes[0].inputs[0]
            ),
        };
        let input_shape = sub_inputs.shape().to_vec();

        let sub_outputs_str = nodes
            .iter()
            .map(|node| node.outputs.clone())
            .flatten()
            .collect::<Vec<_>>();
        let sub_outputs = match tensor_map.get(&nodes[nodes.len() - 1].outputs[0]) {
            Some(x) => x,
            None => panic!(
                "Error occurs at test_cnn: tensor '{}' not found",
                &nodes[nodes.len() - 1].outputs[0]
            ),
        };
        let output_shape = sub_outputs.shape().to_vec();

        let sub_tensor_map: BTreeMap<_, _> = tensor_map
            .clone()
            .into_iter()
            .filter(|(key, _)| sub_inputs_str.contains(key) || sub_outputs_str.contains(key))
            .collect();

        (
            Self {
                tensor_map: sub_tensor_map,
                nodes: nodes,
                input_shape,
                output_shape,
            },
            sub_outputs.clone(),
        )
    }

    pub fn construct_sub_backward_graph(
        nodes: Vec<Node>,
        tensor_map: BTreeMap<String, Tensor>,
    ) -> (Self, Tensor) {
        let sub_inputs_str = nodes
            .iter()
            .map(|node| node.backward_inputs.clone())
            .flatten()
            .collect::<Vec<_>>();
        let sub_inputs = match tensor_map.get(&nodes[nodes.len() - 1].backward_inputs[0]) {
            Some(x) => x,
            None => panic!(
                "Error occurs at test_cnn: tensor '{}' not found",
                &nodes[nodes.len() - 1].backward_inputs[0]
            ),
        };
        let input_shape = sub_inputs.shape().to_vec();

        let sub_outputs_str = nodes
            .iter()
            .map(|node| node.backward_outputs.clone())
            .flatten()
            .collect::<Vec<_>>();
        let sub_outputs = match tensor_map.get(&nodes[0].backward_outputs[0]) {
            Some(x) => x,
            None => panic!(
                "Error occurs at test_cnn: tensor '{}' not found",
                &nodes[0].backward_outputs[0]
            ),
        };
        let output_shape = sub_outputs.shape().to_vec();

        let sub_tensor_map: BTreeMap<_, _> = tensor_map
            .clone()
            .into_iter()
            .filter(|(key, _)| sub_inputs_str.contains(key) || sub_outputs_str.contains(key))
            .collect();

        (
            Self {
                tensor_map: sub_tensor_map,
                nodes: nodes,
                input_shape,
                output_shape,
            },
            sub_outputs.clone(),
        )
    }
}

// #[derive(Clone, Debug)]
// pub struct GraphInput {
//     pub data: Tensor,
//     pub label: Int,
// }
//
// impl GraphInput {
//     pub fn construct(inputs: Vec<Input>, scale_factor: u64) -> Vec<Self> {
//         inputs
//             .iter()
//             .map(|input| Self {
//                 data: Tensor::from_shape_vec(
//                     IxDyn(&[1, input.data.len()]),
//                     input
//                         .data
//                         .iter()
//                         .map(|x| ((x * scale_factor as f64) as Int))
//                         .collect(),
//                 )
//                 .unwrap(),
//                 label: input.label as Int,
//             })
//             .collect()
//     }
// }
