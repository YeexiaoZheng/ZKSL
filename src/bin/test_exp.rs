use std::collections::BTreeSet;

use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
    circuits::exp_circuit::ExpCircuit,
    utils::helpers::{configure_static_numeric_config, to_field},
};

fn main() {
    // original vector
    let v_input1: Vec<i64> = vec![10; 3];
    let v_input2: Vec<i64> = vec![3; 3];
    let v_output: Vec<i64> = vec![3; 3];
    // [10, 10, 10] / [3, 3, 3] = [3, 3, 3]

    // field vector
    let f_input1 = v_input1
        .iter()
        .map(|x| to_field::<Fr>(*x))
        .collect::<Vec<_>>();
    let f_input2 = v_input2
        .iter()
        .map(|x| to_field::<Fr>(*x))
        .collect::<Vec<_>>();
    let f_output = v_output
        .iter()
        .map(|x| to_field::<Fr>(*x))
        .collect::<Vec<_>>();

    let circuit = ExpCircuit::construct(f_input1, f_input2);

    let k = 10;
    let scale_factor = 1;

    configure_static_numeric_config(k, 12, scale_factor, BTreeSet::new());

    let prover = MockProver::run(k as u32, &circuit, vec![f_output]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
