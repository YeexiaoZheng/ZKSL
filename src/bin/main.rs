use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
    graph::Graph,
    model::ModelCircuit,
    utils::{
        helpers::{configure_static_numeric_config, to_field},
        loader::load_from_json,
    },
};

fn main() {
    // Load graph
    let scale_factor = 8;
    let graph = Graph::construct(load_from_json("src/utils/test.json"), scale_factor);
    println!("{:?}", graph);
    let circuit = ModelCircuit::<Fr>::construct(graph);

    // Set numeric config
    let k = 14;
    configure_static_numeric_config(k, 12, scale_factor, circuit.clone().used_numerics.clone());

    // Run the circuit
    let output = circuit.forward().unwrap();
    println!("output: {:?}", output);

    // Verify the circuit
    let public = output.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
    let prover = MockProver::run(k as u32, &circuit, vec![public]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}
