use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use ndarray::{Array, Dim, IxDyn};
use zkml::layers::fully_connected::FullyConnectedCircuit;

fn to_field(x: i64) -> Fr {
    let bias = 1 << 31;
    let x_pos = x + bias;
    Fr::from(x_pos as u64) - Fr::from(bias as u64)
}

fn main() {
    // original vector
    let v_input: Vec<i64> = vec![1; 10];
    let v_hidden_layer: Vec<i64> = vec![1; 100];

    // original matrix
    let o_input = Array::<i64, Dim<_>>::from_shape_vec([1, 10], v_input.clone()).unwrap();
    let o_hidden_layer =
        Array::<i64, Dim<_>>::from_shape_vec([10, 10], v_hidden_layer.clone()).unwrap();
    let o_output = o_input.dot(&o_hidden_layer);
    println!("{:?}", o_input);
    println!("{:?}", o_hidden_layer);
    println!("{:?}", o_output);

    // field matrix
    let f_input = Array::from_shape_vec(
        IxDyn(o_input.shape()),
        v_input.iter().map(|x| to_field(*x)).collect::<Vec<_>>(),
    )
    .unwrap();
    let f_hidden_layer = Array::from_shape_vec(
        IxDyn(o_hidden_layer.shape()),
        v_hidden_layer
            .iter()
            .map(|x| to_field(*x))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let f_output = Array::from_shape_vec(
        IxDyn(o_output.shape()),
        o_output.iter().map(|x| to_field(*x)).collect::<Vec<_>>(),
    )
    .unwrap();

    // let input = f_input.clone().into_iter().collect::<Vec<_>>();
    let output = f_output.clone().into_iter().collect::<Vec<_>>();

    let circuit = FullyConnectedCircuit::construct(f_input, f_hidden_layer);

    let prover = MockProver::run(10, &circuit, vec![output]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
