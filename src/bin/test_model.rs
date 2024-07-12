use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
    commitment::poseidon::PoseidonHash,
    graph::Graph,
    model::ModelCircuit,
    utils::{
        helpers::{configure_static_numeric_config, to_field},
        loader::load_from_json,
    },
    weight::FieldWeight,
};

type F = Fr;

fn main() {
    // Load graph
    let scale_factor = 512;
    let graph = Graph::construct(load_from_json("jsons/mlp2.json"), scale_factor);
    // println!("{:?}", graph);
    let circuit = ModelCircuit::<F>::construct(graph.clone());

    // Set numeric config
    let k = 15;
    configure_static_numeric_config(
        k,
        12,
        scale_factor,
        1,
        circuit.clone().used_numerics.clone(),
    );

    // Run the circuit
    let output = circuit.forward().unwrap();
    println!("output: {:?}", output);
    for (i, o) in output.iter().enumerate() {
        println!("output[{}]: {}({})", i, o, *o as f64 / scale_factor as f64);
    }

    let weight = FieldWeight::<F>::construct(graph.nodes.clone(), circuit.field_tensor_map.clone());
    let hash_output = PoseidonHash::<F>::hash_vec(weight.to_vec());
    // println!("hash_output: {:?}", hash_output);

    // Verify the circuit
    let mut public = output.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
    public.extend(hash_output);
    // println!("public: {:?}", public);
    let prover = MockProver::run(k as u32, &circuit, vec![public]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}
