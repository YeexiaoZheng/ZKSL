use std::collections::BTreeMap;

use clap::Parser;
use log::info;
use ndarray::{concatenate, Axis};
use rayon::ThreadPoolBuilder;
use zkdeepfm::{
    graph::Graph,
    net::client::FederatedClient,
    numeric::NumericConfig,
    stage::{Trainer, NO_PROVE},
    utils::{
        args::Cli,
        helpers::{
            configure_static, configure_static_numeric_config_default, get_numeric_config,
            set_log_level, to_field,
        },
        loader::load_from_json,
    },
};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let parallel = cli.parallel;
    unsafe {
        NO_PROVE = cli.no_prove;
    }

    let batch_size = cli.batch_size;

    // Logging
    set_log_level(log::LevelFilter::Info);

    // Network
    let mut client = FederatedClient::connect("http://[::1]:50051".to_string()).await;

    // Parameters
    let scale_factor = 512;
    let mut forward_k = 17;
    let mut backward_k = 18;
    let forward_assigned_num_cols = 1;
    let backward_assigned_num_cols = 2;
    let mut num_cols = 10;
    let rlr = 1;
    let epoch = 3;

    if parallel {
        forward_k = 16;
        backward_k = 17;
        num_cols = 6;
    }

    // Config numeric config
    let numeric_config = configure_static(NumericConfig {
        num_cols,
        scale_factor,
        batch_size,
        reciprocal_learning_rate: rlr,
        random_size: 1000 * batch_size,
        use_selectors: true,
        ..configure_static_numeric_config_default()
    });

    // Load graph
    let graph = Graph::construct(
        load_from_json("examples/lenet_conv/model.json"),
        numeric_config.scale_factor,
        false,
    );
    // println!("{:#?}", graph);

    // let cores = core_affinity::get_core_ids().unwrap();
    // println!("Available CPU cores: {:?}", cores);
    // let num_threads = cores.len().min(32);
    // let num_offset = 32;

    // let pool = ThreadPoolBuilder::new()
    //     .num_threads(num_threads)
    //     .start_handler(move |index| {
    //         if let Some(core_id) = cores.get(index % cores.len() + num_offset) {
    //             core_affinity::set_for_current(*core_id);
    //             // println!("Thread {} assigned to CPU core {:?}", index, core_id.id);
    //         }
    //     })
    //     .build()
    //     .unwrap();
    let pool = ThreadPoolBuilder::new().build().unwrap();

    let mut trainer = Trainer::new(
        forward_k,
        forward_k, // gradient_k not needed
        backward_k,
        graph.clone(),
        None,
        pool,
        Some("params/federated_lenet/conv".to_string()),
    );
    let mut s_proofs = vec![];

    for e in 0..epoch {
        info!("----- epoch: {:?} -----", e);
        let epoch_start = std::time::Instant::now();
        let start = std::time::Instant::now();

        let input = graph.tensor_map.get("input").unwrap().clone();
        let input = (0..batch_size).map(|_| input.view()).collect::<Vec<_>>();
        let input = concatenate(Axis(0), &input).unwrap();
        println!("input shape: {:?}", input.shape());
        let input = input.mapv(|x| to_field(x));
        let input = BTreeMap::from([("input".to_string(), input)]);

        let scores = if parallel {
            trainer.forward_parallel(input.clone())
        } else {
            trainer.forward(input.clone())
        };

        info!("forward compute time: {:?}", start.elapsed());
        let start = std::time::Instant::now();

        // Client send data to server and get proof
        let proofs = client
            .send_data(("input".to_string(), scores.clone()))
            .await
            .unwrap();
        println!("proofs len: {:?}", proofs.len());

        info!("send data time: {:?}", start.elapsed());
        let start = std::time::Instant::now();

        // Prove forward circuit
        configure_static(NumericConfig {
            assigned_num_cols: forward_assigned_num_cols,
            ..get_numeric_config()
        });
        let proofs = trainer.prove_forward();
        s_proofs.extend(proofs);

        info!("prove forward time: {:?}", start.elapsed());
        let start = std::time::Instant::now();

        // Client send proof to server and get gradient
        let gradient = client
            .send_proof(("input".to_string(), s_proofs.clone()))
            .await
            .unwrap();
        s_proofs.clear();

        info!("send proof time: {:?}", start.elapsed());
        let start = std::time::Instant::now();

        /* Run backward circuit */
        let _ = if parallel {
            trainer.backward_parallel(gradient.clone())
        } else {
            trainer.backward(gradient.clone())
        };

        info!("backward compute time: {:?}", start.elapsed());
        let start = std::time::Instant::now();

        // Prove backward circuit
        configure_static(NumericConfig {
            assigned_num_cols: backward_assigned_num_cols,
            ..get_numeric_config()
        });
        let proofs = trainer.prove_backward();
        s_proofs.extend(proofs);

        info!("prove backward time: {:?}", start.elapsed());
        // let res = run_verify_kzg(pk.get_vk(), gradient, proof.commitment.clone(), proof);
        // info!("KZG Backward Circuit Verified: {}", res);

        info!("----- epoch cost time: {:?} -----", epoch_start.elapsed());
    }
}
