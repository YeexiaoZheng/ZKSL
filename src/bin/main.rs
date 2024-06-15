use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

use ndarray::{Array, Dim, IxDyn};
use zkml::{graph::Graph, model::{FormatLayer, ModelCircuit}, utils::loader::load_from_json};

fn to_field(x: i64) -> Fr {
    let bias = 1 << 31;
    let x_pos = x + bias;
    Fr::from(x_pos as u64) - Fr::from(bias as u64)
}

fn main() {
    // // original vector
    // let v_input: Vec<i64> = vec![1, 2];
    // let v_hidden_layer: Vec<i64> = vec![1, 2, 3, 4];

    // // original matrix
    // let o_input = Array::<i64, Dim<_>>::from_shape_vec([1, 2], v_input.clone()).unwrap();
    // let o_hidden_layer =
    //     Array::<i64, Dim<_>>::from_shape_vec([2, 2], v_hidden_layer.clone()).unwrap();
    // let o_output = o_input.dot(&o_hidden_layer);
    // println!("{:?}", o_input);
    // println!("{:?}", o_hidden_layer);
    // println!("{:?}", o_output);

    // // field matrix
    // let f_input = Array::from_shape_vec(
    //     IxDyn(o_input.shape()),
    //     v_input.iter().map(|x| to_field(*x)).collect::<Vec<_>>(),
    // )
    // .unwrap();
    // let f_hidden_layer = Array::from_shape_vec(
    //     IxDyn(o_hidden_layer.shape()),
    //     v_hidden_layer
    //         .iter()
    //         .map(|x| to_field(*x))
    //         .collect::<Vec<_>>(),
    // )
    // .unwrap();
    // let f_output = Array::from_shape_vec(
    //     IxDyn(o_output.shape()),
    //     o_output.iter().map(|x| to_field(*x)).collect::<Vec<_>>(),
    // )
    // .unwrap();

    // let input = f_input.clone().into_iter().collect::<Vec<_>>();
    // let output = f_output.clone().into_iter().collect::<Vec<_>>();

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
