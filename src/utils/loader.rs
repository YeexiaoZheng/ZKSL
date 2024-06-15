use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorJson {
    pub shape: Vec<usize>,
    pub data: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeJson {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphJson {
    pub tensor_map: HashMap<String, TensorJson>,
    pub nodes: Vec<NodeJson>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

pub fn load_from_json(_file_path: &str) -> GraphJson {
    let tmp_json = r#"
    {
        "tensor_map": {
            "input": {
                "shape": [1, 10], 
                "data": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            },
            "mlp1.weight": {
                "shape": [10, 10], 
                "data": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            },
            "mlp2.weight": {
                "shape": [1, 2],
                "data": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            }
        },
        "nodes": [
            {
                "op_type": "Gemm",
                "inputs": ["input", "mlp1.weight", "mlp1.bias"],
                "outputs": ["/mlp1/Gemm_output_0"],
                "attributes": {"alpha": (1.0, 1), "beta": (1.0, 1), "transB": (0.0, 2)}
            },
            {
                "op_type": "Gemm",
                "inputs": ["/mlp1/Gemm_output_0", "mlp2.weight", "mlp2.bias"],
                "outputs": ["output"],
                "attributes": {"alpha": (1.0, 1), "beta": (1.0, 1), "transB": (0.0, 2)}
            }
        ],
        "input_shape": [1, 10],
        "output_shape": [1, 2]
    }
    "#;
    serde_json::from_str(tmp_json).unwrap()
}
