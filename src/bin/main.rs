use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

use ndarray::{Array, Dim, IxDyn};
use zkml::{graph::Graph, model::{FormatLayer, ModelCircuit}, utils::loader::load_from_json};

fn main() {
    // let k = 3;
    // let circuit = ModelCircuit::<Fr>::construct(
    //     k,
    //     vec![FormatLayer {
    //         layer_name: "FullyConnected".to_string(),
    //         input_shape: vec![1, 2],
    //         output_shape: vec![1, 2],
    //         weight_shape: vec![2, 2],
    //         original_weights: o_hidden_layer,
    //         field_weights: f_hidden_layer,
    //     }],
    // );

    // load graph
    let graph = Graph::construct(load_from_json("file path"));

    let circuit = ModelCircuit::<Fr>::construct(3, graph);

    // MockProver::run(3, &circuit, vec![input, output]).unwrap();
}
