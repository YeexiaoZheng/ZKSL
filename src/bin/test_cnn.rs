use std::collections::BTreeMap;

use log::info;
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

const FKZG: bool = true;
// const FKZG: bool = false;
const GKZG: bool = true;
// const GKZG: bool = false;
const BKZG: bool = true;
// const BKZG: bool = false;

const FORWARD_K: u32 = 17;
const GRADIENT_K: u32 = 15;
const BACKWARD_K: u32 = 18;

fn main() {
    // Logging
    set_log_level(log::LevelFilter::Info);

    // Parameters
    let scale_factor = 1024;
    let forward_assigned_num_cols = 1;
    let gradient_assigned_num_cols = 1;
    let backward_assigned_num_cols = 2;
    let num_cols = 9;
    let rlr = 1;
    let epoch = 3;

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
        graph,
        Some(LossType::SoftMax),
        pool,
        Some("params/lenet".to_string()),
    );

    for e in 0..epoch {
        info!("----- epoch: {:?} -----", e);
        let start = std::time::Instant::now();

        /* Run forward circuit */
        let input = trainer.graph.tensor_map.get("input").unwrap().clone();
        let input = input.mapv(|x| to_field(x));

        // let backward = trainer.train(input, vec![1]);
        let input = BTreeMap::from([("input".to_string(), input)]);
        let scores = trainer.forward(input);
        let gradient = trainer.gradient(scores.clone(), vec![1]);
        let _ = trainer.backward(gradient.clone());

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

        info!("----- epoch cost time: {:?} -----", start.elapsed());
    }
}
