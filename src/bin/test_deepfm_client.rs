use log::info;
use rayon::ThreadPoolBuilder;
use std::{collections::BTreeMap, env};
use zkdeepfm::{
    graph::Graph,
    net::client::FederatedClient,
    numeric::NumericConfig,
    stage::Trainer,
    utils::{
        helpers::{
            configure_static, configure_static_numeric_config_default, set_log_level, to_field,
        },
        loader::load_from_json,
    },
};

const PARALLEL: bool = false;
const IP: &str = "[::1]:50053";

const FORWARD_K: u32 = 17;
const GRADIENT_K: u32 = 12;
const BACKWARD_K: u32 = 18;

#[tokio::main]
async fn main() {
    // Logging
    set_log_level(log::LevelFilter::Info);

    // Network
    let mut client = FederatedClient::connect(IP.to_string()).await;

    let args: Vec<String> = env::args().collect();
    let arg = args
        .get(1)
        .expect("Please provide a command line argument: 1, 2, or 3");
    let arg_value: u32 = arg.parse().expect("Argument must be a number: 1, 2, or 3");

    // Parameters
    let scale_factor = 512;
    let mut k = 18;
    let mut num_cols = 10;
    let mut assigned_num_cols = 1;
    let rlr = 1;
    let epoch = 3;

    if PARALLEL {
        k = 17;
        num_cols = 6;
        assigned_num_cols = 2;
    }

    // Config numeric config
    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        batch_size: 1,
        reciprocal_learning_rate: rlr,
        assigned_num_cols,
        use_selectors: true,
        ..configure_static_numeric_config_default()
    });

    // Load graph
    let graph = Graph::construct(
        load_from_json(&format!("examples/deepfm_client_{}/model.json", arg_value)),
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
        None,
        pool,
        Some(format!("params/federated_deepfm/client_{}", arg_value)),
    );
    let mut s_proofs = vec![];

    for e in 0..epoch {
        info!("----- epoch: {:?} -----", e);
        let start = std::time::Instant::now();

        let input = graph.tensor_map.get("input").unwrap().clone();
        let input = input.mapv(|x| to_field(x));
        let input = BTreeMap::from([(format!("input_{}", arg_value), input)]);

        let scores = if PARALLEL {
            trainer.forward_parallel(input.clone())
        } else {
            trainer.forward(input.clone())
        };

        // Client send data to server and get proof
        let proofs = client
            .send_data((format!("client_{}", arg_value), scores.clone()))
            .await
            .unwrap();
        println!("proofs len: {:?}", proofs.len());

        // Prove forward circuit
        let proofs = trainer.prove_forward();
        s_proofs.extend(proofs);

        // Client send proof to server and get gradient
        let gradient = client
            .send_proof((format!("client_{}", arg_value), s_proofs.clone()))
            .await
            .unwrap();
        s_proofs.clear();

        /* Run backward circuit */
        let _ = if PARALLEL {
            trainer.backward_parallel(gradient.clone())
        } else {
            trainer.backward(gradient.clone())
        };

        let proofs = trainer.prove_backward();
        s_proofs.extend(proofs);
        // let res = run_verify_kzg(pk.get_vk(), gradient, proof.commitment.clone(), proof);
        // info!("KZG Backward Circuit Verified: {}", res);

        info!("----- epoch cost time: {:?} -----", start.elapsed());
    }
}
