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
    let scale_factor = 512;
    let graph = Graph::construct(load_from_json("jsons/mlp2.json"), scale_factor);
    // println!("{:?}", graph);
    let circuit = ModelCircuit::<Fr>::construct(graph);

    // Set numeric config
    let k = 15;
    configure_static_numeric_config(k, 12, scale_factor, circuit.clone().used_numerics.clone());

    // Run the circuit
    let output = circuit.forward().unwrap();
    println!("output: {:?}", output);
    for (i, o) in output.iter().enumerate() {
        println!("output[{}]: {}({})", i, o, *o as f64 / scale_factor as f64);
    }

    // Verify the circuit
    let public = output.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
    let prover = MockProver::run(k as u32, &circuit, vec![public]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}
