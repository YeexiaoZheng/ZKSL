use std::{collections::BTreeSet, vec};

use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

use zkml::{
    graph::Graph,
    loss::loss::LossType,
    stage::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit},
    utils::{
        helpers::{configure_static_numeric_config, to_field},
        loader::load_from_json,
    },
};

type F = Fr;

fn main() {
    // Parameters
    let scale_factor = 1024;
    let k = 15;
    let num_cols = 12;
    configure_static_numeric_config(k, num_cols, scale_factor, BTreeSet::new());

    // Load graph
    let graph = Graph::construct(load_from_json("jsons/mlp2.json"), scale_factor);
    // println!("{:?}", graph);

    // Run forward circuit
    let input = graph.tensor_map.get("input").unwrap().clone();
    let forward_circuit = ForwardCircuit::<Fr>::construct(graph.clone());
    let score = forward_circuit.run(&input).unwrap();
    println!("score: {:?}", score);

    // Run gradient circuit
    let label = vec![0];
    let gradient_circuit =
        GradientCircuit::<F>::construct(score.clone(), label.clone(), LossType::SoftMax);
    let (loss, gradient) = gradient_circuit.run().unwrap();
    println!("loss: {:?}", loss);
    println!("gradient: {:?}", gradient);

    // Run backward circuit
    // let backward_circuit = BackwardCircuit::<Fr>::construct(graph.clone());
    // let backward_gradient = backward_circuit.run(&gradient).unwrap();

    // Set numeric config
    configure_static_numeric_config(
        k,
        num_cols,
        scale_factor,
        forward_circuit.clone().used_numerics.clone(),
    );
    // Verify the circuit
    let forward_public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
    let forward_prover = MockProver::run(k as u32, &forward_circuit, vec![forward_public]).unwrap();
    assert_eq!(forward_prover.verify(), Ok(()));

    // Set numeric config
    configure_static_numeric_config(
        k,
        num_cols,
        scale_factor,
        gradient_circuit.clone().used_numerics.clone(),
    );
    // Verify the circuit
    let gradient_public = gradient.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
    let gradient_prover =
        MockProver::run(k as u32, &gradient_circuit, vec![gradient_public]).unwrap();
    assert_eq!(gradient_prover.verify(), Ok(()));

    // Set numeric config
    // configure_static_numeric_config(
    //     k,
    //     num_cols,
    //     scale_factor,
    //     backward_circuit.clone().used_numerics.clone(),
    // );
    // // Verify the circuit
    // let backward_public = backward_gradient
    //     .iter()
    //     .map(|x| to_field(*x))
    //     .collect::<Vec<_>>();
    // let backward_prover =
    //     MockProver::run(k as u32, &backward_circuit, vec![backward_public]).unwrap();
    // assert_eq!(backward_prover.verify(), Ok(()));
}
