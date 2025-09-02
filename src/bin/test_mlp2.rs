use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

use log::info;
use zkdeepfm::{
    graph::Graph,
    loss::LossType,
    numeric::NumericConfig,
    prover::prover::ProverKZG,
    stage::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit},
    utils::{
        helpers::{
            configure_static, configure_static_numeric_config_default, set_log_level, to_field,
            update_graph,
        },
        loader::load_from_json,
    },
};

type F = Fr;

const FKZG: bool = true;
const GKZG: bool = true;
const BKZG: bool = true;

fn main() {
    // Logging
    set_log_level(log::LevelFilter::Info);

    // Parameters
    let scale_factor = 1024;
    let k = 14;
    let num_cols = 12;
    let rlr = 1;
    let epoch = 2;

    // Config numeric config
    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        batch_size: 1,
        reciprocal_learning_rate: rlr,
        ..configure_static_numeric_config_default()
    });

    // Load graph
    let mut graph = Graph::construct(
        load_from_json("examples/mlp2/model.json"),
        numeric_config.scale_factor,
        true,
    );
    // println!("{:#?}", graph);

    for e in 0..epoch {
        info!("----- epoch: {:?} -----", e);
        let start = std::time::Instant::now();

        /* Run forward circuit */
        let _input = graph.tensor_map.get("input").unwrap().clone();
        let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone());
        let score = forward_circuit.run().unwrap();
        info!("score: {}", score);

        // Verify the circuit
        let forward_public = score.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
        info!("forward_public: {:?}", forward_public);
        let forward_prover =
            MockProver::run(k, &forward_circuit, vec![forward_public.clone()]).unwrap();
        assert_eq!(forward_prover.verify(), Ok(()));
        info!("Mock Forward Circuit Verified");

        if FKZG {
            let mut forward_prover = ProverKZG::construct(k, forward_circuit.clone(), None);
            forward_prover.load();
            let _proof = forward_prover.prove(
                vec![forward_public.clone()],
                numeric_config.assigned_num_cols,
                forward_circuit.commitment_tuples(),
            );
            // let res = forward_prover.verify(proof.clone(), proof.commitment.clone());
            // info!("KZG Forward Circuit Verified: {}", res);
        }

        /* Run gradient circuit */
        let label = vec![1];
        let gradient_circuit =
            GradientCircuit::<F>::construct(score.clone(), label.clone(), LossType::SoftMax);
        let (loss, gradient) = gradient_circuit.run().unwrap();
        info!("loss: {}, gradient: {}", loss, gradient);

        // Verify the circuit
        let gradient_public = gradient
            .iter()
            .map(|x| to_field::<F>(*x))
            .collect::<Vec<_>>();
        let gradient_prover =
            MockProver::run(k, &gradient_circuit, vec![gradient_public.clone()]).unwrap();
        assert_eq!(gradient_prover.verify(), Ok(()));
        info!("Mock Gradient Circuit Verified");

        if GKZG {
            let mut gradient_prover = ProverKZG::construct(k, gradient_circuit.clone(), None);
            gradient_prover.load();
            let _proof = gradient_prover.prove(
                vec![gradient_public.clone()],
                numeric_config.assigned_num_cols,
                vec![],
            );
            // let res = gradient_prover.verify(proof.clone(), proof.commitment.clone());
            // info!("KZG Gradient Circuit Verified: {}", res);
        }

        /* Run backward circuit */
        let mut backward_graph = forward_circuit.graph.clone();
        backward_graph
            .tensor_map
            .insert("gradient".to_string(), gradient.clone());
        let mut backward_circuit = BackwardCircuit::<F>::construct(backward_graph.clone());
        let backward_gradient = backward_circuit.run().unwrap();
        info!("backward_gradient: {}", backward_gradient);
        // println!("graph: {:#?}", backward_circuit.graph);

        // Verify the circuit
        let backward_public = backward_gradient
            .iter()
            .map(|x| to_field(*x))
            .collect::<Vec<_>>();
        let backward_prover =
            MockProver::run(k as u32, &backward_circuit, vec![backward_public.clone()]).unwrap();
        assert_eq!(backward_prover.verify(), Ok(()));
        info!("Mock Backward Circuit Verified");

        if BKZG {
            let mut backward_prover = ProverKZG::construct(k, backward_circuit.clone(), None);
            backward_prover.load();
            let _proof = backward_prover.prove(
                vec![backward_public.clone()],
                numeric_config.assigned_num_cols,
                backward_circuit.commitment_tuples(),
            );
            // let res = backward_prover.verify(proof.clone(), proof.commitment.clone());
            // info!("KZG Backward Circuit Verified: {}", res);
        }

        // Update graph (mainly the weights)
        graph = update_graph(&graph, &backward_circuit.graph.tensor_map);
        info!("----- epoch cost time: {:?} -----", start.elapsed());
    }
}
