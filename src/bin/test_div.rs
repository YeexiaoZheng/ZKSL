
use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zksl::{
    circuits::div_circuit::DivCircuit,
    utils::{
        helpers::{configure_static_numeric_config_default, to_field},
        math::Int,
    },
};

fn main() {
    // original vector
    let v_input1: Vec<Int> = vec![10; 3];
    let v_input2: Vec<Int> = vec![3; 3];
    let v_output: Vec<Int> = vec![3; 3];
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

    let circuit = DivCircuit::construct(f_input1, f_input2);

    let numeric_config = configure_static_numeric_config_default();

    let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_output]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
