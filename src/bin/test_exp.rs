
use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zksl::{
    circuits::exp_circuit::ExpCircuit,
    utils::{
        helpers::{configure_static_numeric_config_default, to_field},
        math::{exp, Int},
    },
};

fn main() {
    let numeric_config = configure_static_numeric_config_default();

    // original vector
    let v_input: Vec<Int> = vec![0, 1, 2, 3, 4, 100];
    let v_output: Vec<Int> = v_input.iter().map(|x| exp(*x, numeric_config.scale_factor)).collect();
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

    let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_output]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
