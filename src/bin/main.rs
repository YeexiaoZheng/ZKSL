use std::sync::Arc;

use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
    graph::Graph,
    model::ModelCircuit,
    numerics::numeric::NumericConfig,
    utils::{
        helpers::{to_field, NUMERIC_CONFIG},
        loader::load_from_json,
    },
};

// use zkml::{graph::Graph, model::ModelCircuit, utils::loader::load_from_json};

fn main() {
    // Load graph
    let graph = Graph::construct(load_from_json("src/utils/test.json"));
    println!("{:?}", graph);
    let circuit = ModelCircuit::<Fr>::construct(graph);

    // Set numeric config
    let k = 10;
    let scale_factor = 1 << 9;
    let nconfig = &NUMERIC_CONFIG;
    let cloned = nconfig.lock().unwrap().clone();
    *nconfig.lock().unwrap() = NumericConfig {
        k,
        scale_factor,
        num_rows: (1 << k) - 10 + 1,
        num_cols: 10,
        use_selectors: true,
        used_numerics: Arc::new(circuit.clone().used_numerics),
        ..cloned
    };

    // Run the circuit
    let output = circuit.forward().unwrap();
    println!("{:?}", output);

    // Verify the circuit
    let public = output.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
    let prover = MockProver::run(10, &circuit, vec![public]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}
