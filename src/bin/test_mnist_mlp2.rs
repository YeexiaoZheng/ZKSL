use halo2_proofs::{
    halo2curves::bn256::{Fr, G1Affine},
    plonk::{ProvingKey, VerifyingKey},
};

use ndarray::Array;
use zksl::{
    commitments::poseidon::PoseidonHash,
    graph::{Graph, GraphInput},
    loss::LossType,
    numeric::NumericConfig,
    prover::prover_kzg::{KZGProver, StageType},
    stages::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit},
    utils::{
        helpers::{configure_static, to_field, update_graph, Tensor},
        loader::{load_from_json, load_input_from_json},
    },
    weight::{FieldWeight, Weight},
};

type F = Fr;

fn main() {
    // Parameters
    let scale_factor = 1024;
    let k = 20;
    let num_cols = 12;
    let batch_size = 10;
    let lr = 100;
    let epoch = 10;
    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        num_rows: 1 << k - 10 + 1,
        max_val: 1 << 17 - 1,
        min_val: -(1 << 17),
        scale_factor,
        batch_size,
        learning_rate: lr,
        commitment: true,
        ..Default::default()
    });

    // Load graph
    let mut graph = Graph::construct(
        load_from_json("examples/mnist_mlp2/model.json"),
        scale_factor,
    );
    // Load inputs
    let inputs = GraphInput::construct(
        load_input_from_json("examples/mnist_mlp2/input.json"),
        scale_factor,
    );
    // println!("{:#?}", graph);

    let mut prover = KZGProver::construct(k);
    let mut fpk: Option<ProvingKey<G1Affine>> = Default::default();
    let mut fvk: Option<VerifyingKey<G1Affine>> = Default::default();
    let mut gpk: Option<ProvingKey<G1Affine>> = Default::default();
    let mut gvk: Option<VerifyingKey<G1Affine>> = Default::default();
    let mut bpk: Option<ProvingKey<G1Affine>> = Default::default();
    let mut bvk: Option<VerifyingKey<G1Affine>> = Default::default();

    for e in 0..epoch {
        println!("----- epoch: {:?} -----", e);
        let start = std::time::Instant::now();

        // Only compute the hash output once
        let forward_circuit = ForwardCircuit::<F>::construct(graph.clone(), &numeric_config);
        let hash_output = PoseidonHash::<F>::hash_vec_to_one(
            // Poseidon hash the forward weights
            FieldWeight::<F>::construct(
                forward_circuit.graph.nodes.clone(),
                forward_circuit.field_tensor_map.clone(),
            )
            .to_vec(),
        );

        let input = Array::from_shape_vec(
            (inputs.len(), inputs[0].data.len()),
            inputs
                .clone()
                .into_iter()
                .map(|inp| inp.data.into_iter())
                .flatten()
                .collect(),
        )
        .unwrap()
        .into_dyn();

        /* Run forward circuit */
        // let _input = graph.tensor_map.get("input").unwrap().clone();
        let x = graph
            .tensor_map
            .entry("input".to_string())
            .or_insert(input.clone());
        *x = input.clone();
        let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone(), &numeric_config);
        let score = forward_circuit.run().unwrap();
        println!(
            "label: \n{:?}, \nscore: \n{:?}",
            inputs.iter().map(|inp| inp.label).collect::<Vec<_>>(),
            score
        );
        let _ = configure_static(NumericConfig {
            // Set numeric config
            used_numerics: forward_circuit.clone().used_numerics.clone().into(),
            ..numeric_config.clone()
        });
        // Set public
        let mut forward_public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
        forward_public.push(hash_output);
        // keygen_forward
        if prover.stage.forward.is_none() {
            prover.set_forward(forward_circuit.clone());
            let (pk, vk) = prover.keygen_forward();
            (fpk, fvk) = (Some(pk).into(), Some(vk).into());
        }
        // Create proof
        let proof = prover.prove(
            StageType::Forward,
            &fpk.clone().unwrap(),
            forward_public.clone(),
        );
        println!("sizeof(proof): {} Bytes", std::mem::size_of_val(&proof));
        // Verify the proof
        assert!(prover.verify(
            StageType::Forward,
            &fvk.clone().unwrap(),
            forward_public,
            proof
        ));
        println!("Forward circuit verified");

        /* Run gradient circuit */
        let label = inputs.iter().map(|x| x.label).collect::<Vec<_>>();
        let gradient_circuit =
            GradientCircuit::<F>::construct(score.clone(), label.clone(), LossType::SoftMax);
        let (loss, gradient) = gradient_circuit.run().unwrap();
        println!("loss: \n{:?}, \ngradient: \n{:?}", loss, gradient);
        let numeric_config = configure_static(NumericConfig {
            // Set numeric config
            batch_size,
            used_numerics: gradient_circuit.clone().used_numerics.clone().into(),
            ..numeric_config.clone()
        });
        // Set public
        let gradient_public = gradient
            .iter()
            .map(|x| to_field::<F>(*x))
            .collect::<Vec<_>>();
        // keygen_gradient
        // if prover.stage.gradient.is_none() {
        //     prover.set_gradient(gradient_circuit.clone());
        //     let (pk, vk) = prover.keygen_gradient();
        //     (gpk, gvk) = (Some(pk).into(), Some(vk).into());
        // }
        // // Create proof
        // let proof = prover.prove(
        //     StageType::Gradient,
        //     &gpk.clone().unwrap(),
        //     gradient_public.clone(),
        // );
        // println!("sizeof(proof): {} Bytes", std::mem::size_of_val(&proof));
        // // Verify the proof
        // assert!(prover.verify(
        //     StageType::Gradient,
        //     &gvk.clone().unwrap(),
        //     gradient_public,
        //     proof
        // ));
        // println!("Gradient circuit verified");

        /* Run backward circuit */
        let mut backward_graph = forward_circuit.graph.clone();
        backward_graph
            .tensor_map
            .insert("gradient".to_string(), gradient.clone());
        let mut backward_circuit =
            BackwardCircuit::<F>::construct(backward_graph.clone(), &numeric_config);
        let backward_gradient = backward_circuit.run().unwrap();
        println!("backward_gradient: \n{:?}", backward_gradient);
        // println!("graph: {:#?}", backward_circuit.graph);

        let _ = configure_static(NumericConfig {
            // Set numeric config
            learning_rate: lr,
            used_numerics: backward_circuit.clone().used_numerics.clone().into(),
            ..numeric_config
        });
        // Poseidon hash the forward weights & backward weights
        let forward_hash_output = PoseidonHash::<F>::hash_vec_to_one(
            FieldWeight::<F>::construct(
                backward_circuit.graph.nodes.clone(),
                backward_circuit.field_tensor_map.clone(),
            )
            .to_vec(),
        );
        let backward_hash_output = PoseidonHash::<F>::hash_vec_to_one(
            FieldWeight::<F>::construct_from_weight(Weight::construct(
                backward_circuit.graph.nodes.clone(),
                backward_circuit.graph.tensor_map.clone(),
            ))
            .to_vec(),
        );
        // Verify the circuit
        let mut backward_public = backward_gradient
            .iter()
            .map(|x| to_field(*x))
            .collect::<Vec<_>>();
        backward_public.push(forward_hash_output);
        backward_public.push(backward_hash_output);
        // keygen_backward
        if prover.stage.backward.is_none() {
            prover.set_backward(backward_circuit.clone());
            let (pk, vk) = prover.keygen_backward();
            (bpk, bvk) = (Some(pk).into(), Some(vk).into());
        }
        // Create proof
        let proof = prover.prove(
            StageType::Backward,
            &bpk.clone().unwrap(),
            backward_public.clone(),
        );
        println!("sizeof(proof): {} Bytes", std::mem::size_of_val(&proof));
        // Verify the proof
        assert!(prover.verify(
            StageType::Backward,
            &bvk.clone().unwrap(),
            backward_public,
            proof
        ));
        println!("Backward circuit verified");

        // Update graph (mainly the weights)
        graph = update_graph(&graph, &backward_circuit.graph.tensor_map);
        println!("----- epoch cost time: {:?} -----", start.elapsed());
    }
}
