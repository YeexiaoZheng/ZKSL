use std::collections::BTreeSet;

use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
    circuits::exp_circuit::ExpCircuit,
    utils::helpers::{configure_static_numeric_config, to_field},
};

fn main() {
    // original vector
    let v_input: Vec<i64> = vec![10; 1];
    let v_output: Vec<i64> = vec![10; 1];
    println!("{}", 1 << 17);

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

    let k = 6;
    let scale_factor = 1;

    configure_static_numeric_config(k, 2, scale_factor, BTreeSet::new());

    let prover = MockProver::run(k as u32, &circuit, vec![f_output]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
