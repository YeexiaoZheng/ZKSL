use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use ndarray::Array;
use zkml::{
    loss::loss::LossType,
    stage::gradient::GradientCircuit,
    utils::helpers::{configure_static_numeric_config, to_field},
};

type F = Fr;

fn main() {
    // Load
    let scale_factor = 512;
    let k = 15;

    let score = Array::from_shape_vec((1, 2), vec![300, 700])
        .unwrap()
        .into_dyn();
    let label = vec![0];
    let circuit = GradientCircuit::<F>::construct(score, label, LossType::SoftMax);

    // Set numeric config
    configure_static_numeric_config(k, 12, scale_factor, circuit.clone().used_numerics.clone());

    // Run the circuit
    let (loss, gradient) = circuit.run().unwrap();
    println!("loss: {:?}", loss);
    println!("gradient: {:?}", gradient);
    // for (i, o) in gradient.iter().enumerate() {
    //     println!("gradient[{}]: {}({})", i, o, *o as f64 / scale_factor as f64);
    // }

    // Verify the circuit
    let public = gradient.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
    let prover = MockProver::run(k as u32, &circuit, vec![public]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}
