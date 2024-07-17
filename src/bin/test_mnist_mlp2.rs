use halo2_proofs::{dev::MockProver, halo2curves::pasta::Fp};

use ndarray::{Array, IxDyn};
use zksl::{
    commitments::poseidon::PoseidonHash,
    graph::{Graph, GraphInput},
    loss::loss::LossType,
    numerics::numeric::NumericConfig,
    stages::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit},
    utils::{
        helpers::{configure_static, to_field, update_graph},
        loader::{load_from_json, load_input_from_json},
    },
    weight::{FieldWeight, Weight},
};

type F = Fp;

fn main() {
    // Parameters
    let scale_factor = 1024;
    let k = 16;
    let num_cols = 12;
    let batch_size = 10;
    let lr = 100;
    let epoch = 20;
    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        batch_size,
        learning_rate: lr,
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

    for e in 0..epoch {
        println!("----- epoch: {:?} -----", e);
        let start = std::time::Instant::now();

        // Only compute the hash output once
        let forward_circuit = ForwardCircuit::<F>::construct(graph.clone());
        let hash_output = PoseidonHash::<F>::hash_vec_to_one(
            // Poseidon hash the forward weights
            FieldWeight::<F>::construct(
                forward_circuit.graph.nodes.clone(),
                forward_circuit.field_tensor_map.clone(),
            )
            .to_vec(),
        );

        let mut scores = vec![];
        for input in inputs.iter() {
            /* Run forward circuit */
            // let _input = graph.tensor_map.get("input").unwrap().clone();
            let x = graph
                .tensor_map
                .entry("input".to_string())
                .or_insert(input.data.clone());
            *x = input.data.clone();
            let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone());
            let score = forward_circuit.run().unwrap();
            scores.push(score.clone());
            println!("label: {}, score: {:?}", input.label, score);
            let _ = configure_static(NumericConfig {
                // Set numeric config
                used_numerics: forward_circuit.clone().used_numerics.clone().into(),
                ..numeric_config.clone()
            });

            // Verify the circuit
            let mut forward_public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
            forward_public.push(hash_output);
            let forward_prover =
                MockProver::run(k as u32, &forward_circuit, vec![forward_public]).unwrap();
            assert_eq!(forward_prover.verify(), Ok(()));
            println!("Forward circuit verified");
        }

        /* Run gradient circuit */
        let label = inputs.iter().map(|x| x.label).collect::<Vec<_>>();
        let score = Array::from_shape_vec(
            IxDyn(&[batch_size, scores[0].len()]),
            scores.iter().flatten().cloned().collect(),
        )
        .unwrap();
        let gradient_circuit =
            GradientCircuit::<F>::construct(score.clone(), label.clone(), LossType::SoftMax);
        let (loss, gradient) = gradient_circuit.run().unwrap();
        println!("loss: {:?}, gradient: {:?}", loss, gradient);
        let numeric_config = configure_static(NumericConfig {
            // Set numeric config
            batch_size,
            used_numerics: gradient_circuit.clone().used_numerics.clone().into(),
            ..numeric_config.clone()
        });
        // Verify the circuit
        let gradient_public = gradient.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
        let gradient_prover =
            MockProver::run(k as u32, &gradient_circuit, vec![gradient_public]).unwrap();
        assert_eq!(gradient_prover.verify(), Ok(()));
        println!("Gradient circuit verified");

        /* Run backward circuit */
        let mut backward_graph = graph.clone();
        backward_graph
            .tensor_map
            .insert("gradient".to_string(), gradient.clone());
        let mut backward_circuit = BackwardCircuit::<F>::construct(backward_graph.clone(), lr);
        let backward_gradient = backward_circuit.run().unwrap();
        println!("backward_gradient: {:?}", backward_gradient);
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
        let backward_prover =
            MockProver::run(k as u32, &backward_circuit, vec![backward_public]).unwrap();
        assert_eq!(backward_prover.verify(), Ok(()));
        println!("Backward circuit verified");

        // Update graph (mainly the weights)
        graph = update_graph(&graph, &backward_circuit.graph.tensor_map);
        println!("----- epoch cost time: {:?} -----", start.elapsed());
    }
}
