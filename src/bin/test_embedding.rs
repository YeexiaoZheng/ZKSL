use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

use zkdeepfm::{
    graph::Graph,
    loss::LossType,
    numeric::NumericConfig,
    prover::prover::ProverKZG,
    stage::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit},
    utils::{
        helpers::{
            configure_static, configure_static_numeric_config_default, set_log_level, to_field,
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
    let num_cols = 6;

    // Config numeric config
    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        batch_size: 1,
        ..configure_static_numeric_config_default()
    });

    // Load graph
    let graph = Graph::construct(
        load_from_json("examples/embedding/model.json"),
        scale_factor,
        false,
    );
    // println!("{:#?}", graph);

    /* Run forward circuit */
    let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone());
    let score = forward_circuit.run().unwrap();
    println!("score: {:?}", score);
    let forward_public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();

    // MockProver
    let forward_prover =
        MockProver::run(k as u32, &forward_circuit, vec![forward_public.clone()]).unwrap();
    assert_eq!(forward_prover.verify(), Ok(()));
    println!("Mock forward circuit verified");

    if FKZG {
        let mut forward_prover = ProverKZG::construct(k as u32, forward_circuit.clone(), None);
        forward_prover.load();
        let _proof = forward_prover.prove(
            vec![forward_public.clone()],
            numeric_config.assigned_num_cols,
            vec![],
        );
        // let res = forward_prover.verify(proof.clone(), proof.commitment.clone());
        // println!("KZG forward circuit verified: {:?}", res);
    }

    /* Run gradient circuit */
    let label = vec![0];
    let gradient_circuit =
        GradientCircuit::<F>::construct(score.clone(), label.clone(), LossType::SoftMax);
    let (loss, gradient) = gradient_circuit.run().unwrap();
    println!("loss: {:?}, gradient: {:?}", loss, gradient);
    let gradient_public = gradient.iter().map(|x| to_field(*x)).collect::<Vec<_>>();

    // MockProver
    let gradient_prover =
        MockProver::run(k as u32, &gradient_circuit, vec![gradient_public.clone()]).unwrap();
    assert_eq!(gradient_prover.verify(), Ok(()));
    println!("Mock gradient circuit verified");

    if GKZG {
        let mut gradient_prover = ProverKZG::construct(k, gradient_circuit.clone(), None);
        gradient_prover.load();
        let _proof = gradient_prover.prove(
            vec![gradient_public.clone()],
            numeric_config.assigned_num_cols,
            vec![],
        );
        // let res = gradient_prover.verify(proof.clone(), proof.commitment.clone());
        // println!("KZG gradient circuit verified: {:?}", res);
    }

    /* Run backward circuit */
    let mut backward_graph = forward_circuit.graph.clone();
    backward_graph
        .tensor_map
        .insert("gradient".to_string(), gradient.clone());
    let mut backward_circuit = BackwardCircuit::<F>::construct(backward_graph.clone());
    let b_gradient = backward_circuit.run().unwrap();
    println!("backward_gradient: {:?}", b_gradient);
    // println!("graph: {:#?}", backward_circuit.graph);
    let backward_public = b_gradient.iter().map(|x| to_field(*x)).collect::<Vec<_>>();

    // MockProver
    let backward_prover =
        MockProver::run(k as u32, &backward_circuit, vec![backward_public.clone()]).unwrap();
    assert_eq!(backward_prover.verify(), Ok(()));
    println!("Mock backward circuit verified");

    if BKZG {
        let mut backward_prover = ProverKZG::construct(k, backward_circuit.clone(), None);
        backward_prover.load();
        let _proof = backward_prover.prove(
            vec![backward_public.clone()],
            numeric_config.assigned_num_cols,
            vec![],
        );
        // let res = backward_prover.verify(proof.clone(), proof.commitment.clone());
        // println!("KZG backward circuit verified: {:?}", res);
    }
}
