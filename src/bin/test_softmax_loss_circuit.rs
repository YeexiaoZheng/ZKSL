use std::collections::BTreeSet;

use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use ndarray::Array;
use zkml::{
    circuits::softmax_loss_circuit::SoftMaxLossCircuit,
    utils::helpers::{configure_static_numeric_config, to_field, NUMERIC_CONFIG},
};

fn main() {
    type F = Fr;
    let k = 16;
    println!("row: {}", (1 << k) - 10 + 1);
    let scale_factor = 1000;
    let num_cols = 6;

    let input = vec![300, 700];
    let input = Array::from_shape_vec([1, 2], input).unwrap().into_dyn();
    let label = vec![0];

    let f_input = input.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_label = label.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_input = Array::from_shape_vec([1, 2], f_input).unwrap().into_dyn();

    configure_static_numeric_config(k, num_cols, scale_factor, BTreeSet::new());

    let circuit = SoftMaxLossCircuit::construct(f_input, f_label);

    let (loss, gradient) = circuit
        .compute(&input, &label, &NUMERIC_CONFIG.lock().unwrap().clone())
        .unwrap();

    println!("loss: {:?}", loss);
    println!("gradient: {:?}", gradient);

    let f_gradient = gradient
        .iter()
        .map(|x| to_field::<F>(*x))
        .collect::<Vec<_>>();
    println!("f_gradient: {:?}", f_gradient);

    let prover = MockProver::run(k as u32, &circuit, vec![f_gradient]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
