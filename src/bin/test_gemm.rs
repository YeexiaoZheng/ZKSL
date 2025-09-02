use std::collections::BTreeMap;
use log::info;
use ndarray::{Array, IxDyn};
use rayon::ThreadPoolBuilder;
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

const FORWARD_K: u32 = 21;

fn main() {
    // Logging
    set_log_level(log::LevelFilter::Info);

    // Parameters
    let scale_factor = 1024;
    let forward_assigned_num_cols = 3;
    let num_cols = 9;
    let rlr = 1;
    let epoch = 1;

    // Config numeric config
    let numeric_config = configure_static(NumericConfig {
        // k,
        num_cols,
        scale_factor,
        batch_size: 1,
        reciprocal_learning_rate: rlr,
        use_selectors: true,
        ..configure_static_numeric_config_default()
    });

    // Load graph
    let graph = Graph::construct(
        load_from_json("examples/mlp2/gemm.json"),
        numeric_config.scale_factor,
        true,
    );

    let pool = ThreadPoolBuilder::new().build().unwrap();

    let mut trainer = Trainer::new(
        FORWARD_K,
        14,
        14,
        graph,
        Some(LossType::SoftMax),
        pool,
        Some("params/lenet".to_string()),
    );

    for e in 0..epoch {
        info!("----- epoch: {:?} -----", e);

        /* Run forward circuit */
        let i = vec![1024; 64*20000];
        let i = Array::from_shape_vec(IxDyn(&[64,20000]), i).unwrap();
        trainer.graph.tensor_map.insert("input".to_string(), i.clone());

        let w = vec![1024; 20000*9];
        let w = Array::from_shape_vec(IxDyn(&[20000,9]), w).unwrap();
        trainer.graph.tensor_map.insert("mlp1.weight".to_string(), w.clone());

        let b = vec![1024; 9];
        let b = Array::from_shape_vec(IxDyn(&[9]), b).unwrap();
        trainer.graph.tensor_map.insert("mlp1.bias".to_string(), b.clone());

        let ii = trainer.graph.tensor_map.get("input").unwrap();
        let ww = trainer.graph.tensor_map.get("mlp1.weight").unwrap();
        let bb = trainer.graph.tensor_map.get("mlp1.bias").unwrap();
        println!("input: {:?}", ii.shape());
        println!("weight: {:?}", ww.shape());
        println!("bias: {:?}", bb.shape());

        let input = trainer.graph.tensor_map.get("input").unwrap().clone();
        let input = input.mapv(|x| to_field(x));

        let input = BTreeMap::from([("input".to_string(), input)]);
        let _scores = trainer.forward(input);

        configure_static(NumericConfig {
            assigned_num_cols: forward_assigned_num_cols,
            ..numeric_config.clone()
        });

        let start = std::time::Instant::now();
        let proofs = trainer.prove_forward();
        println!("gemm time: {:?}", start.elapsed());

        println!("{:?}", proofs[0].commitment);
        info!("----- epoch cost time: {:?} -----", start.elapsed());
    }
}
