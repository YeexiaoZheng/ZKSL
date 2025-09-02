use std::{fs::File, io::BufReader};

use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr, Map};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorJson {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeJson {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub backward_inputs: Vec<String>,
    pub backward_outputs: Vec<String>,
    #[serde_as(as = "Map<DisplayFromStr, _>")]
    pub attributes: Vec<(String, Vec<f64>)>,
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

pub fn load_from_json(file_path: &str) -> GraphJson {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        panic!("Failed to load json! ")
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputJson {
    pub shape: Vec<usize>,
    pub data: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Input {
    pub input: InputJson,
    pub label: InputJson,
    pub inputs: Vec<InputJson>,
}

pub fn load_input_from_json(file_path: &str) -> Input {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    match serde_json::from_reader(reader) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("Error: {}", e);
            panic!("Failed to load input from json! ")
        }
    }
}
