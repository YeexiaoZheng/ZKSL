use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr, Map};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorJson {
    pub shape: Vec<usize>,
    pub data: Vec<i64>,
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeJson {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde_as(as = "Map<DisplayFromStr, _>")]
    pub attributes: Vec<(String, f64)>,
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphJson {
    #[serde_as(as = "Map<DisplayFromStr, _>")]
    pub tensor_map: Vec<(String, TensorJson)>,
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
                "shape": [10, 2],
                "data": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            }
        },
        "nodes": [
            {
                "op_type": "Gemm",
                "inputs": ["input", "mlp1.weight"],
                "outputs": ["/mlp1/Gemm_output_0"],
                "attributes": {"alpha": 1.0, "beta": 1.0, "transB": 1}
            },
            {
                "op_type": "Gemm",
                "inputs": ["/mlp1/Gemm_output_0", "mlp2.weight"],
                "outputs": ["output"],
                "attributes": {"alpha": 1.0, "beta": 1.0, "transB": 1}
            }
        ],
        "input_shape": [1, 10],
        "output_shape": [1, 2]
    }
    "#;
    match serde_json::from_str(tmp_json) {
        Ok(graph_json) => graph_json,
        Err(e) => {
            eprintln!("Error: {}", e);
            panic!("Failed to load graph from json! ")
        }
    }
}
