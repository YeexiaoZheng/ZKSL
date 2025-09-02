use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use std::fs;

use log::info;
use ndarray::Array;
use zkdeepfm::utils::loader::load_input_from_json;
use zkdeepfm::utils::math::Int;
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
    let k = 20;
    let num_cols = 10;
    let epoch = 2;

    // Config numeric config
    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        assigned_num_cols: 3,
        batch_size: 32,
        feature_num: 31,
        ..configure_static_numeric_config_default()
    });

    // Load graph
    let mut graph = Graph::construct(
        load_from_json("examples/loan_default/model_2.json"),
        numeric_config.scale_factor,
        false,
    );
    // println!("{:#?}", graph);

    let batch_files_path = std::path::Path::new("examples/loan_default/batch_files");
    let num_batches = fs::read_dir(batch_files_path)
        .expect("Failed to read batch_files directory")
        .count();

    for e in 0..epoch {
        info!("----- epoch: {:?} -----", e);

        let start = std::time::Instant::now();

        for b in 0..num_batches {
            let start = std::time::Instant::now();

            // Construct file path for current batch
            let file_path = batch_files_path.join(format!("batch_{}.json", b));
            let input_data = load_input_from_json(file_path.to_str().unwrap());

            // Update input in graph's tensor map
            let input = Array::from_shape_vec(input_data.input.shape, input_data.input.data)
                .unwrap()
                .into_dyn();

            let tensor_input_ref = graph
                .tensor_map
                .entry("input".to_string())
                .or_insert(input.clone());
            *tensor_input_ref = input.clone();

            for (idx, tensor_json) in input_data.inputs.iter().enumerate() {
                let input_key = format!("input{}", idx);
                let tensor =
                    Array::from_shape_vec(tensor_json.shape.clone(), tensor_json.data.clone())
                        .unwrap()
                        .into_dyn();
                let tensor_ref = graph.tensor_map.entry(input_key).or_insert(tensor.clone());
                *tensor_ref = tensor.clone();
            }

            /* Run forward circuit */
            let _ = configure_static(NumericConfig {
                k,
                ..numeric_config.clone()
            });
            let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone());
            let score = forward_circuit.run().unwrap();
            // let score_shape = score.shape();
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
                let proof = forward_prover.prove(
                    vec![forward_public.clone()],
                    numeric_config.assigned_num_cols,
                    forward_circuit.commitment_tuples(),
                );
                let res = forward_prover.verify(
                    proof.clone(),
                    vec![forward_public],
                    proof.commitment.clone(),
                );
                info!("KZG Forward Circuit Verified: {}", res);
            }

            /* Run gradient circuit */
            let _ = configure_static(NumericConfig {
                k,
                ..numeric_config.clone()
            });
            let label = input_data.label.data;

            let _zero: Int = 0;
            // let s_min = score.clone().iter().fold(zero, |acc, &s| if s < 0 { s } else { acc });
            // let s_max = score.clone().iter().fold(zero, |acc, &s| if s > 0 { s } else { acc });
            let s_min = score.iter().min().unwrap().clone();
            let s_max = score.iter().max().unwrap().clone();
            let score = score.mapv(|x| x - s_min);
            let score = score.mapv(|x| {
                ((x as f64 / (s_max as f64 - s_min as f64)) * numeric_config.scale_factor as f64)
                    as Int
            });
            info!("normalized score: {}", score);

            let gradient_circuit =
                GradientCircuit::<F>::construct(score.clone(), label.clone(), LossType::Sigmoid);
            let (loss, gradient) = gradient_circuit.run().unwrap();
            info!("loss: {}, gradient: {}", loss, gradient);

            // Verify the circuit
            let gradient_public = gradient.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
            let gradient_prover =
                MockProver::run(k, &gradient_circuit, vec![gradient_public.clone()]).unwrap();
            assert_eq!(gradient_prover.verify(), Ok(()));
            info!("Mock Gradient Circuit Verified");

            if GKZG {
                let mut gradient_prover = ProverKZG::construct(k, gradient_circuit.clone(), None);
                gradient_prover.load();
                let proof = gradient_prover.prove(
                    vec![gradient_public.clone()],
                    numeric_config.assigned_num_cols,
                    vec![],
                );
                let res = gradient_prover.verify(
                    proof.clone(),
                    vec![gradient_public],
                    proof.commitment.clone(),
                );
                info!("KZG Gradient Circuit Verified: {}", res);
            }

            /* Run backward circuit */
            let _ = configure_static(NumericConfig {
                k,
                ..numeric_config.clone()
            });
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
                MockProver::run(k as u32, &backward_circuit, vec![backward_public.clone()])
                    .unwrap();
            assert_eq!(backward_prover.verify(), Ok(()));
            info!("Mock Backward Circuit Verified");

            if BKZG {
                let mut backward_prover = ProverKZG::construct(k, backward_circuit.clone(), None);
                backward_prover.load();
                let proof = backward_prover.prove(
                    vec![backward_public.clone()],
                    numeric_config.assigned_num_cols,
                    backward_circuit.commitment_tuples(),
                );
                let res = backward_prover.verify(
                    proof.clone(),
                    vec![backward_public],
                    proof.commitment.clone(),
                );
                info!("KZG Backward Circuit Verified: {}", res);
            }

            // Update graph (mainly the weights)
            graph = update_graph(&graph, &backward_circuit.graph.tensor_map);

            info!("----- batch cost time: {:?} -----", start.elapsed());
        }
        info!("----- epoch cost time: {:?} -----", start.elapsed());
    }
}
