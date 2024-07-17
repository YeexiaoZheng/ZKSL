use std::sync::Arc;

use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zksl::{
    commitments::poseidon::PoseidonHash,
    graph::Graph,
    model::ModelCircuit,
    numerics::numeric::NumericConfig,
    utils::{
        helpers::{configure_static, configure_static_numeric_config_default, to_field},
        loader::load_from_json,
    },
    weight::FieldWeight,
};

type F = Fr;

fn main() {
    let numeric_config = configure_static_numeric_config_default();

    // Load graph
    let graph = Graph::construct(
        load_from_json("jsons/mlp2.json"),
        numeric_config.scale_factor,
    );
    // println!("{:?}", graph);
    let circuit = ModelCircuit::<F>::construct(graph.clone());

    // Set numeric config
    let numeric_config = configure_static(NumericConfig {
        used_numerics: Arc::new(circuit.clone().used_numerics.clone()),
        ..numeric_config
    });

    // Run the circuit
    let output = circuit.forward().unwrap();
    println!("output: {:?}", output);
    for (i, o) in output.iter().enumerate() {
        println!(
            "output[{}]: {}({})",
            i,
            o,
            *o as f64 / numeric_config.scale_factor as f64
        );
    }

    let weight = FieldWeight::<F>::construct(graph.nodes.clone(), circuit.field_tensor_map.clone());
    let hash_output = PoseidonHash::<F>::hash_vec(weight.to_vec());
    // println!("hash_output: {:?}", hash_output);

    // Verify the circuit
    let mut public = output.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
    public.extend(hash_output);
    // println!("public: {:?}", public);
    let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![public]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}
