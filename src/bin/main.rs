use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

use ndarray::{Array, Dim, IxDyn};
use zkml::model::{FormatLayer, ModelCircuit};

fn to_field(x: i64) -> Fr {
    let bias = 1 << 31;
    let x_pos = x + bias;
    Fr::from(x_pos as u64) - Fr::from(bias as u64)
}

fn main() {
    // original matrix
    let o_input = Array::<i64, Dim<_>>::from_shape_vec((1, 2), vec![1, 2]).unwrap();
    let o_hidden_layer = Array::<i64, Dim<_>>::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
    let o_output = o_input.dot(&o_hidden_layer);
    println!("{:?}", o_input);
    println!("{:?}", o_hidden_layer);
    println!("{:?}", o_output);

    // field matrix
    // Array::
    o_input.
    let input = Array::from_shape_vec(IxDyn(o_input.shape()), );
    let input = Array::from_shape_vec(
        IxDyn(&vec![1, 2]),
        vec![1, 2].iter().map(|x| to_field(*x)).collect::<Vec<_>>(),
    )
    .unwrap();
    let hidden_layer = Array::from_shape_vec(
        IxDyn(&vec![2, 2]),
        vec![1, 2, 3, 4]
            .iter()
            .map(|x| to_field(*x))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let output = vec![7, 10].iter().map(|x| to_field(*x)).collect::<Vec<_>>();

    let hidden_layer = hidden_layer.into_dyn();
    let k = 3;

    let circuit = ModelCircuit::<Fr>::construct(
        k,
        vec![FormatLayer {
            name: "FullyConnected".to_string(),
            input_shape: vec![1, 2],
            output_shape: vec![1, 2],
            weight_shape: vec![2, 2],
            original_weights: o_hidden_layer,
            field_weights: hidden_layer,
        }],
    );

    MockProver::run(3, &circuit, vec![output]).unwrap();
}
