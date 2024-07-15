use std::{collections::BTreeSet, vec};

use halo2_proofs::{
    dev::MockProver,
    halo2curves::{bn256::Fr, pasta::Fp},
};

use zkml::{
    commitments::poseidon::PoseidonHash,
    graph::Graph,
    loss::loss::LossType,
    stages::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit},
    utils::{
        helpers::{configure_static_numeric_config, to_field, update_graph},
        loader::load_from_json,
    },
    weight::{FieldWeight, Weight},
};

type F = Fr;

fn main() {
    // Parameters
    let scale_factor = 1024;
    let k = 15;
    let num_cols = 12;
    let lr = 1;
    let epoch = 20;
    configure_static_numeric_config(k, num_cols, scale_factor, 1, BTreeSet::new());

    // Load graph
    let mut graph = Graph::construct(load_from_json("jsons/mlp2.json"), scale_factor);
    // println!("{:#?}", graph);

    for _ in 0..epoch {
        // Run forward circuit
        let _input = graph.tensor_map.get("input").unwrap().clone();
        let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone());
        let score = forward_circuit.run().unwrap();
        println!("score: {:?}", score);

        // Run gradient circuit
        let label = vec![1];
        let gradient_circuit =
            GradientCircuit::<F>::construct(score.clone(), label.clone(), LossType::SoftMax);
        let (loss, gradient) = gradient_circuit.run().unwrap();
        println!("loss: {:?}, gradient: {:?}", loss, gradient);

        // Run backward circuit
        let mut backward_graph = forward_circuit.graph.clone();
        backward_graph
            .tensor_map
            .insert("gradient".to_string(), gradient.clone());
        let mut backward_circuit = BackwardCircuit::<F>::construct(backward_graph.clone(), lr);
        let backward_gradient = backward_circuit.run().unwrap();
        println!("backward_gradient: {:?}", backward_gradient);
        // println!("graph: {:#?}", backward_circuit.graph);

        // Update graph (mainly the weights)
        graph = update_graph(&graph, &backward_circuit.graph.tensor_map);

        // Set numeric config
        configure_static_numeric_config(
            k,
            num_cols,
            scale_factor,
            1,
            forward_circuit.clone().used_numerics.clone(),
        );

        // Poseidon hash the forward weights
        let weight = FieldWeight::<F>::construct(
            forward_circuit.graph.nodes.clone(),
            forward_circuit.field_tensor_map.clone(),
        );
        let hash_output = PoseidonHash::<F>::hash_vec(weight.to_vec());

        // Verify the circuit
        let mut forward_public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
        forward_public.extend(hash_output);
        let forward_prover =
            MockProver::run(k as u32, &forward_circuit, vec![forward_public]).unwrap();
        assert_eq!(forward_prover.verify(), Ok(()));

        println!("Forward circuit verified");

        // Set numeric config
        configure_static_numeric_config(
            k,
            num_cols,
            scale_factor,
            1,
            gradient_circuit.clone().used_numerics.clone(),
        );
        // Verify the circuit
        let gradient_public = gradient.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
        let gradient_prover =
            MockProver::run(k as u32, &gradient_circuit, vec![gradient_public]).unwrap();
        assert_eq!(gradient_prover.verify(), Ok(()));

        println!("Gradient circuit verified");

        // Set numeric config
        configure_static_numeric_config(
            k,
            num_cols,
            scale_factor,
            1,
            backward_circuit.clone().used_numerics.clone(),
        );

        // Poseidon hash the forward weights & backward weights
        let forward_weight = FieldWeight::<F>::construct(
            backward_circuit.graph.nodes.clone(),
            backward_circuit.field_tensor_map.clone(),
        );
        let forward_hash_output = PoseidonHash::<F>::hash_vec(forward_weight.to_vec());
        let backward_weight = Weight::construct(
            backward_circuit.graph.nodes.clone(),
            backward_circuit.graph.tensor_map.clone(),
        );
        let backward_weight = FieldWeight::<F>::construct_from_weight(backward_weight);
        let backward_hash_output = PoseidonHash::<F>::hash_vec(backward_weight.to_vec());

        // Verify the circuit
        let mut backward_public = backward_gradient
            .iter()
            .map(|x| to_field(*x))
            .collect::<Vec<_>>();
        backward_public.extend(forward_hash_output);
        backward_public.extend(backward_hash_output);
        let backward_prover =
            MockProver::run(k as u32, &backward_circuit, vec![backward_public]).unwrap();
        assert_eq!(backward_prover.verify(), Ok(()));

        println!("Backward circuit verified");
    }
}
