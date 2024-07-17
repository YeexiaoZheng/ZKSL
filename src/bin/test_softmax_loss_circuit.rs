use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use ndarray::Array;
use zksl::{
    circuits::softmax_loss_circuit::SoftMaxLossCircuit,
    utils::helpers::{configure_static_numeric_config_default, to_field},
};

type F = Fr;

fn main() {
    let input = vec![300, 700];
    let input = Array::from_shape_vec([1, 2], input).unwrap().into_dyn();
    let label = vec![0];

    let f_input = input.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_label = label.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_input = Array::from_shape_vec([1, 2], f_input).unwrap().into_dyn();

    let numeric_config = configure_static_numeric_config_default();

    let circuit = SoftMaxLossCircuit::construct(f_input, f_label);

    let (loss, gradient) = circuit.compute(&input, &label, &numeric_config).unwrap();

    println!("loss: {:?}", loss);
    println!("gradient: {:?}", gradient);

    let f_gradient = gradient
        .iter()
        .map(|x| to_field::<F>(*x))
        .collect::<Vec<_>>();

    let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_gradient]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
