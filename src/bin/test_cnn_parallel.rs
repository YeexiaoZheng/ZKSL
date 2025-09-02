use halo2_proofs::halo2curves::bn256::Fr;
use rayon::ThreadPoolBuilder;
use std::collections::BTreeMap;

use log::info;
use zkdeepfm::{
    graph::Graph,
    loss::LossType,
    numeric::NumericConfig,
    stage::Trainer,
    utils::{
        helpers::{
            configure_static, configure_static_numeric_config_default, set_log_level, to_field,
        },
        loader::load_from_json,
    },
};

type F = Fr;

// const ASSIGNED_COLUMN_SIZE: usize = 2;

const FKZG: bool = true;
// const FKZG: bool = false;
const GKZG: bool = true;
// const GKZG: bool = false;
const BKZG: bool = true;
// const BKZG: bool = false;

const FORWARD_K: u32 = 16;
const GRADIENT_K: u32 = 15;
const BACKWARD_K: u32 = 17;

fn main() {
    // Logging
    set_log_level(log::LevelFilter::Info);

    // Parameters
    let scale_factor = 1024;
    let forward_assigned_num_cols = 1;
    let gradient_assigned_num_cols = 1;
    let backward_assigned_num_cols = 2;
    let k = 17;
    let num_cols = 6;
    let rlr = 1;
    let epoch = 1;

    // Config numeric config
    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        batch_size: 1,
        assigned_num_cols: 1,
        reciprocal_learning_rate: rlr,
        use_selectors: true,
        ..configure_static_numeric_config_default()
    });

    // Load graph
    let graph = Graph::construct(
        // load_from_json("examples/lenet/model.json"),
        load_from_json("examples/lenet/model_lenet.json"),
        numeric_config.scale_factor,
        false,
    );
    // println!("{:#?}", graph);

    let pool = ThreadPoolBuilder::new().build().unwrap();

    let mut trainer = Trainer::new(
        FORWARD_K,
        GRADIENT_K,
        BACKWARD_K,
        graph.clone(),
        Some(LossType::SoftMax),
        pool,
        Some("params/lenet_parallel".to_string()),
    );

    for e in 0..epoch {
        info!("----- epoch: {:?} -----", e);
        let start = std::time::Instant::now();

        /* Run forward circuit */
        // region
        let input = graph.tensor_map.get("input").unwrap().clone();
        let input = input.mapv(|x| to_field::<F>(x));
        let labels = vec![1];
        trainer.train_parallel(BTreeMap::from_iter([("input".to_string(), input)]), labels);

        if FKZG {
            configure_static(NumericConfig {
                assigned_num_cols: forward_assigned_num_cols,
                ..numeric_config.clone()
            });
            let proofs = trainer.prove_forward();
            println!("{:?}", proofs[0].commitment);
        }

        if GKZG {
            configure_static(NumericConfig {
                assigned_num_cols: gradient_assigned_num_cols,
                ..numeric_config.clone()
            });
            let proofs = trainer.prove_gradient();
            println!("{:?}", proofs[0].commitment);
        }

        if BKZG {
            configure_static(NumericConfig {
                assigned_num_cols: backward_assigned_num_cols,
                ..numeric_config.clone()
            });
            let proofs = trainer.prove_backward();
            println!("{:?}", proofs[0].commitment);
        }

        // let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone());
        // let score = forward_circuit.run().unwrap();
        // info!("score: {}", score);

        // // split graph based on "Conv" & "Gemm"
        // let mut split_graph = vec![];
        // let mut tmp = vec![];
        // for node in graph.nodes.clone() {
        //     if (match_op_type(node.op_type.clone()) == OPType::Conv
        //         || match_op_type(node.op_type.clone()) == OPType::GEMM)
        //         && !tmp.is_empty()
        //     {
        //         split_graph.push(tmp);
        //         tmp = vec![];
        //     }
        //     tmp.push(node);
        // }
        // if !tmp.is_empty() {
        //     split_graph.push(tmp);
        // }

        // // Construct the sub circuits
        // let mut forward_sub_circuits = vec![];
        // for (idx, nodes) in split_graph.iter().enumerate() {
        //     let (sub_graph, sub_outputs) = Graph::construct_sub_forward_graph(
        //         nodes.clone(),
        //         forward_circuit.graph.tensor_map.clone(),
        //     );
        //     let sub_forward_circuit = ForwardCircuit::<F>::construct(sub_graph);

        //     let sub_forward_public = sub_outputs
        //         .iter()
        //         .map(|x| to_field::<F>(*x))
        //         .collect::<Vec<_>>();
        //     let sub_forward_prover =
        //         MockProver::run(k, &sub_forward_circuit, vec![sub_forward_public.clone()]).unwrap();
        //     assert_eq!(sub_forward_prover.verify(), Ok(()));
        //     info!(
        //         "{:?} Mock Forward Circuit Verified",
        //         nodes
        //             .iter()
        //             .map(|node| node.op_type.clone())
        //             .collect::<Vec<_>>()
        //     );

        //     if FKZG {
        //         let path = dir.join("forward").join(idx.to_string());
        //         let mut sub_forward_prover =
        //             ProverKZG::construct(k, sub_forward_circuit.clone(), Some(path));
        //         sub_forward_prover.load();
        //         forward_sub_circuits.push((
        //             sub_forward_circuit.clone(),
        //             sub_forward_public.clone(),
        //             sub_forward_prover.clone(),
        //         ));
        //     }
        // }

        // // Parallel prove
        // let prove_start = std::time::Instant::now();
        // let mut handles = vec![];
        // for (sub_circuit, sub_public, sub_prover) in forward_sub_circuits.into_iter() {
        //     let handle = thread::spawn(move || {
        //         let proof = sub_prover.prove(
        //             vec![sub_public.clone()],
        //             numeric_config.assigned_num_cols,
        //             sub_circuit.commitment_tuples(),
        //         );
        //         let res =
        //             sub_prover.verify(proof.clone(), vec![sub_public], proof.commitment.clone());
        //         info!(
        //             "KZG {:?} Forward Circuit Verified: {}",
        //             sub_circuit
        //                 .graph
        //                 .nodes
        //                 .iter()
        //                 .map(|n| n.op_type.clone())
        //                 .collect::<Vec<_>>(),
        //             res
        //         );
        //     });
        //     handles.push(handle);
        // }
        // for handle in handles {
        //     handle.join().unwrap();
        // }
        // info!(
        //     "----- forward parallel prove time: {:?} -----",
        //     prove_start.elapsed()
        // );
        // info!("----- forward time: {:?} -----", start.elapsed());
        // // endregion

        // /* Run gradient circuit */
        // // region
        // let _ = configure_static(NumericConfig {
        //     k: gradient_k,
        //     ..numeric_config.clone()
        // });
        // let label = vec![1];
        // let gradient_circuit =
        //     GradientCircuit::<F>::construct(score.clone(), label.clone(), LossType::SoftMax);
        // let (loss, gradient) = gradient_circuit.run().unwrap();
        // info!("loss: {}, gradient: {}", loss, gradient);

        // // Verify the circuit
        // let gradient_public = gradient
        //     .iter()
        //     .map(|x| to_field::<F>(*x))
        //     .collect::<Vec<_>>();
        // // let gradient_prover =
        // //     MockProver::run(k, &gradient_circuit, vec![gradient_public.clone()]).unwrap();
        // // assert_eq!(gradient_prover.verify(), Ok(()));
        // // info!("Mock Gradient Circuit Verified");

        // if GKZG {
        //     let path = dir.join("gradient");
        //     let mut gradient_prover = ProverKZG::construct(k, gradient_circuit.clone(), Some(path));
        //     gradient_prover.load();
        //     let proof = gradient_prover.prove(
        //         vec![gradient_public.clone()],
        //         numeric_config.assigned_num_cols,
        //         vec![],
        //     );
        //     let res = gradient_prover.verify(
        //         proof.clone(),
        //         vec![gradient_public],
        //         proof.commitment.clone(),
        //     );
        //     info!("KZG Gradient Circuit Verified: {}", res);
        // }
        // info!("----- gradient time: {:?} -----", start.elapsed());
        // // endregion

        // /* Run backward circuit */
        // // region
        // let _ = configure_static(NumericConfig {
        //     k: backward_k,
        //     assigned_num_cols: 2,
        //     ..numeric_config.clone()
        // });
        // let mut backward_graph = forward_circuit.graph.clone();
        // backward_graph
        //     .tensor_map
        //     .insert("gradient".to_string(), gradient.clone());
        // let mut backward_circuit = BackwardCircuit::<F>::construct(backward_graph.clone());
        // let backward_gradient = backward_circuit.run().unwrap();
        // info!("backward_gradient: {}", backward_gradient);
        // let unchanged_backward_graph =
        //     update_graph(&backward_circuit.graph, &forward_circuit.graph.tensor_map);

        // // split graph based on "Conv" & "Gemm"
        // let mut split_graph = vec![];
        // let mut tmp = vec![];
        // for node in graph.nodes.clone() {
        //     if (match_op_type(node.op_type.clone()) == OPType::Conv
        //         || match_op_type(node.op_type.clone()) == OPType::GEMM)
        //         && !tmp.is_empty()
        //     {
        //         split_graph.push(tmp);
        //         tmp = vec![];
        //     }
        //     tmp.push(node);
        // }
        // if !tmp.is_empty() {
        //     split_graph.push(tmp);
        // }

        // // Construct the sub circuits
        // let mut backward_sub_circuits = vec![];
        // for (idx, nodes) in split_graph.iter().enumerate() {
        //     let (sub_graph, sub_outputs) = Graph::construct_sub_backward_graph(
        //         nodes.clone(),
        //         unchanged_backward_graph.tensor_map.clone(),
        //     );
        //     // println!("backward graph tensor_map size: {:#?}", sub_graph.tensor_map.iter().map(|(_, v)| v.len()).sum::<usize>());
        //     let sub_backward_circuit = BackwardCircuit::<F>::construct(sub_graph);
        //     let sub_backward_public = sub_outputs
        //         .iter()
        //         .map(|x| to_field::<F>(*x))
        //         .collect::<Vec<_>>();
        //     let sub_backward_prover =
        //         MockProver::run(k, &sub_backward_circuit, vec![sub_backward_public.clone()])
        //             .unwrap();
        //     assert_eq!(sub_backward_prover.verify(), Ok(()));
        //     info!(
        //         "{:?} Mock Backward Circuit Verified",
        //         nodes
        //             .iter()
        //             .map(|node| node.op_type.clone())
        //             .collect::<Vec<_>>()
        //     );

        //     if BKZG {
        //         let path = dir.join("backward").join(idx.to_string());
        //         let mut sub_backward_prover =
        //             ProverKZG::construct(k, sub_backward_circuit.clone(), Some(path));
        //         sub_backward_prover.load();
        //         backward_sub_circuits.push((
        //             sub_backward_circuit,
        //             sub_backward_public,
        //             sub_backward_prover,
        //         ));
        //     }
        // }

        // // Parallel prove
        // let prove_start = std::time::Instant::now();
        // let mut handles = vec![];
        // for (sub_circuit, sub_public, sub_prover) in backward_sub_circuits.into_iter() {
        //     let handle = thread::spawn(move || {
        //         let proof = sub_prover.prove(
        //             vec![sub_public.clone()],
        //             numeric_config.assigned_num_cols,
        //             sub_circuit.commitment_tuples(),
        //         );
        //         let res =
        //             sub_prover.verify(proof.clone(), vec![sub_public], proof.commitment.clone());
        //         info!(
        //             "KZG {:?} Backward Circuit Verified: {}",
        //             sub_circuit
        //                 .graph
        //                 .nodes
        //                 .iter()
        //                 .map(|n| n.op_type.clone())
        //                 .collect::<Vec<_>>(),
        //             res
        //         );
        //     });
        //     handles.push(handle);
        // }
        // for handle in handles {
        //     handle.join().unwrap();
        // }
        // info!(
        //     "----- backward parallel prove time: {:?} -----",
        //     prove_start.elapsed()
        // );
        // info!("----- backward time: {:?} -----", start.elapsed());
        // endregion

        // Update graph (mainly the weights)
        // graph = update_graph(&graph, &backward_circuit.graph.tensor_map);

        info!("----- epoch cost time: {:?} -----", start.elapsed());
    }
}
