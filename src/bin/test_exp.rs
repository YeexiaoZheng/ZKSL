use std::collections::BTreeSet;

use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
    circuits::exp_circuit::ExpCircuit,
    utils::{
        helpers::{configure_static_numeric_config, to_field},
        math::exp,
    },
};

fn main() {
    let k = 16;
    let scale_factor = 1000;
    let num_cols = 12;

    // original vector
    let v_input: Vec<i64> = vec![0, 1, 2, 3, 4, 100];
    let v_output: Vec<i64> = v_input.iter().map(|x| exp(*x, scale_factor)).collect();
    println!("v_input: {:?}", v_input);
    println!("v_output: {:?}", v_output);

    // field vector
    let f_input = v_input
        .iter()
        .map(|x| to_field::<Fr>(*x))
        .collect::<Vec<_>>();
    let f_output = v_output
        .iter()
        .map(|x| to_field::<Fr>(*x))
        .collect::<Vec<_>>();

    let circuit = ExpCircuit::construct(f_input);

    configure_static_numeric_config(k, num_cols, scale_factor, BTreeSet::new());

    let prover = MockProver::run(k as u32, &circuit, vec![f_output]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
