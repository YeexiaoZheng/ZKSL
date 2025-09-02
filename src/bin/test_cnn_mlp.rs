use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use clap::Parser;
use log::info;
use tonic::transport::Server;
use zkdeepfm::{
    graph::Graph,
    net::server::{federated::federated_service_server::FederatedServiceServer, FederatedServer},
    numeric::NumericConfig,
    stage::NO_PROVE,
    utils::{
        args::Cli,
        helpers::{configure_static, configure_static_numeric_config_default, set_log_level},
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

    // Parameters
    let scale_factor = 1024;
    // let mut k = 18;
    let mut forward_k = 17;
    let mut gradient_k = 15;
    let mut backward_k = 18;
    let mut num_cols = 10;
    let rlr = 1;

    if parallel {
        // k = 17;
        forward_k = 16;
        gradient_k = 15;
        backward_k = 17;
        num_cols = 6;
    }

    // Config numeric config
    let numeric_config = configure_static(NumericConfig {
        num_cols,
        scale_factor,
        batch_size,
        reciprocal_learning_rate: rlr,
        use_selectors: true,
        ..configure_static_numeric_config_default()
    });

    // Load graph
    let graph = Graph::construct(
        // load_from_json("examples/lenet/model.json"),
        load_from_json("examples/lenet_mlp/model.json"),
        numeric_config.scale_factor,
        true,
    );

    let mut init_proofs = BTreeMap::new();
    init_proofs.insert("input".to_string(), vec![]);

    // FederatedService
    let server = FederatedServer {
        ks: [forward_k, gradient_k, backward_k],
        c_num: 1,
        loss: zkdeepfm::loss::LossType::SoftMax,
        graph: Arc::new(Mutex::new(graph)),
        s_proofs: Arc::new((Mutex::new(init_proofs), Default::default())),
        parallel,
        path: "params/federated_lenet/mlp".to_string(),
        ..Default::default()
    };

    info!("Server listening on [::1]:50051");
    Server::builder()
        .add_service(FederatedServiceServer::new(server))
        .serve("[::1]:50051".parse().unwrap())
        .await
        .unwrap();
}
