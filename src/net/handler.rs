use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
};

use halo2_proofs::halo2curves::bn256::Fr;
use log::info;
use rayon::ThreadPoolBuilder;

use crate::{
    graph::Graph,
    loss::LossType,
    numeric::NumericConfig,
    prover::proof::Proof,
    stage::Trainer,
    utils::helpers::{configure_static, get_numeric_config, to_field, update_graph, FieldTensor},
};

type F = Fr;

const FORWARD_ASSIGNED_NUM_COLS: usize = 1;
const GRADIENT_ASSIGNED_NUM_COLS: usize = 1;
const BACKWARD_ASSIGNED_NUM_COLS: usize = 2;

pub fn handle(
    input: BTreeMap<String, FieldTensor<F>>,
    ks: [u32; 3],
    loss: LossType,
    graph: Arc<Mutex<Graph>>,
    data: Arc<(Mutex<BTreeMap<String, FieldTensor<F>>>, Condvar)>,
    proofs: Arc<(Mutex<BTreeMap<String, Vec<Proof>>>, Condvar)>,
    parallel: bool,
    path: String,
) {
    let mut graph = graph.lock().unwrap();

    // let cores = core_affinity::get_core_ids().unwrap();
    // println!("Available CPU cores: {:?}", cores);
    // let num_threads = cores.len().min(32);

    // let pool = ThreadPoolBuilder::new()
    //     .num_threads(num_threads)
    //     .start_handler(move |index| {
    //         if let Some(core_id) = cores.get(index % cores.len()) {
    //             core_affinity::set_for_current(*core_id);
    //             // println!("Thread {} assigned to CPU core {:?}", index, core_id.id);
    //         }
    //     })
    //     .build()
    //     .unwrap();
    let pool = ThreadPoolBuilder::new().build().unwrap();

    let mut trainer = Trainer::new(
        ks[0],
        ks[1],
        ks[2],
        graph.clone(),
        Some(loss),
        pool,
        Some(path),
    );
    let labels = vec![1; get_numeric_config().batch_size];

    let start = std::time::Instant::now();
    let _ = if parallel {
        trainer.train_parallel(input.clone(), labels)
    } else {
        trainer.train(input.clone(), labels)
    };
    info!("train compute time: {:?}", start.elapsed());

    // Update data
    let (lock, cvar) = &*data;
    let mut data = lock.lock().unwrap();
    for (k, _) in input.iter() {
        let b_k = "backward_".to_string() + &k;
        data.insert(
            k.to_string(),
            trainer
                .graph
                .tensor_map
                .get(&b_k)
                .unwrap()
                .clone()
                .mapv(|x| to_field(x)),
        );
    }
    drop(data);
    cvar.notify_all();

    // Update graph (mainly the weights)
    let mut original_graph = graph.clone();
    let _ = match &trainer.b_stage {
        Some(b_stages) => {
            for (circuit, _, _) in b_stages.iter() {
                original_graph = update_graph(&original_graph, &circuit.graph.tensor_map);
            }
        }
        None => panic!("No backward stage found"),
    };
    *graph = original_graph;

    // Prove circuit
    let start = std::time::Instant::now();
    let mut new_proofs = vec![];
    configure_static(NumericConfig {
        assigned_num_cols: FORWARD_ASSIGNED_NUM_COLS,
        ..get_numeric_config()
    });
    new_proofs.extend(trainer.prove_forward());
    info!("prove forward time: {:?}", start.elapsed());
    let start = std::time::Instant::now();
    configure_static(NumericConfig {
        assigned_num_cols: GRADIENT_ASSIGNED_NUM_COLS,
        ..get_numeric_config()
    });
    new_proofs.extend(trainer.prove_gradient());
    info!("prove gradient time: {:?}", start.elapsed());
    let start = std::time::Instant::now();
    configure_static(NumericConfig {
        assigned_num_cols: BACKWARD_ASSIGNED_NUM_COLS,
        ..get_numeric_config()
    });
    new_proofs.extend(trainer.prove_backward());
    info!("prove backward time: {:?}", start.elapsed());

    // Update proofs
    let (lock, cvar) = &*proofs;
    let mut proofs = lock.lock().unwrap();
    for (k, _) in input.iter() {
        proofs.insert(k.to_string(), new_proofs.clone());
    }
    // *proofs = Some(new_proofs);
    cvar.notify_all();
}
