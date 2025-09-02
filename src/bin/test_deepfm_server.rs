use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use log::info;
use tonic::transport::Server;
use zkdeepfm::{
    graph::Graph,
    net::server::{federated::federated_service_server::FederatedServiceServer, FederatedServer},
    numeric::NumericConfig,
    utils::{
        helpers::{configure_static, configure_static_numeric_config_default, set_log_level},
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

    // Parameters
    let scale_factor = 1024;
    let mut k = 18;
    let mut num_cols = 10;
    let mut assigned_num_cols = 1;
    let rlr = 1;

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
        load_from_json("examples/deepfm_server/model.json"),
        numeric_config.scale_factor,
        false,
    );

    let mut init_proofs = BTreeMap::new();
    init_proofs.insert("client_1".to_string(), vec![]);
    init_proofs.insert("client_2".to_string(), vec![]);
    init_proofs.insert("client_3".to_string(), vec![]);

    // FederatedService
    let server = FederatedServer {
        ks: [FORWARD_K, GRADIENT_K, BACKWARD_K],
        c_num: 3,
        loss: zkdeepfm::loss::LossType::Sigmoid,
        graph: Arc::new(Mutex::new(graph)),
        s_proofs: Arc::new((Mutex::new(init_proofs), Default::default())),
        parallel: PARALLEL,
        path: "params/federated_deepfm/server".to_string(),
        ..Default::default()
    };

    info!("Server listening on {}", IP);
    Server::builder()
        .add_service(FederatedServiceServer::new(server))
        .serve(IP.parse().unwrap())
        .await
        .unwrap();
}

// 我现在有一个需求，需要使用rust的tonic库实现，需求是有两个联邦A和B，其中B是服务端，A是客户端，然后A会发送两种类型request，分别是data和proof，当A发送给Bdata时，B需要返回AProof，当A发给Bproof时，B需要返回Adata。A是会先做一部分自己的工作生成data，再做一部分工作生成proof，然后A会先把data发给B，等proof生成结束后再发proof给B，然后B是要接收到A的data后才开始工作，会先生成自己的data，此时B需要等待A的proof，接收到A的proof后，再将自己的data发给A，同时开始自己prove的工作，A拿到B的data后再进行工作，生成自己的data，然后将data发给B，此时B接收到A的data的同时需要将自己的proof发给A，以此类推。但是有可能在B接收到A的proof的时候，自身的data还没处理完成，也可能B接收到A的data的时候，自身的proof还没处理完成，因此可能需要把发送给A的data和proof加锁。请仔细理解我的需求，并给出解决方案
