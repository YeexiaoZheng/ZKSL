use std::collections::BTreeMap;

use halo2_proofs::dev::MockProver;
use halo2_proofs::halo2curves::bn256::Fr;
use ndarray::Array;
use zkdeepfm::circuits::fm_circuit::FMCircuit;
use zkdeepfm::numeric::NumericConfig;
use zkdeepfm::utils::helpers::{configure_static, set_log_level, to_field};
use zkdeepfm::utils::math::Int;

fn main() {
    // Logging
    set_log_level(log::LevelFilter::Info);

    type F = Fr;

    // let numeric_config = configure_static_numeric_config_default();
    let scale_factor = 16;
    let k = 15;
    let num_cols = 4;
    let lr = 1;

    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        batch_size: 1,
        use_selectors: true,
        reciprocal_learning_rate: lr,
        ..Default::default()
    });

    // let input = vec![300, 700];
    let input = vec![300, 700, 300, 600, 500];
    let input = input
        .iter()
        .map(|x| x * numeric_config.scale_factor as Int)
        .collect::<Vec<_>>();
    let input = Array::from_shape_vec([1, 5], input).unwrap().into_dyn();
    // let embedding = vec![1, 2, 3, 4, 5, 6];
    let embedding = (1..31).collect::<Vec<_>>();
    let embedding = embedding
        .iter()
        .map(|x| x * numeric_config.scale_factor as Int)
        .collect::<Vec<_>>();

    let embedding = Array::from_shape_vec([1, 5, 6], embedding)
        .unwrap()
        .into_dyn();
    let label = vec![0];

    let f_input = input.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_embedding = embedding
        .iter()
        .map(|x| to_field::<F>(*x))
        .collect::<Vec<_>>();
    let f_label = label.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_input = Array::from_shape_vec([1, 5], f_input).unwrap().into_dyn();
    let f_embedding = Array::from_shape_vec([1, 5, 6], f_embedding)
        .unwrap()
        .into_dyn();

    let circuit = FMCircuit::construct(f_input, f_embedding, f_label);

    let res = circuit
        .forward(&vec![input, embedding], &numeric_config, &BTreeMap::new())
        .unwrap();

    println!("res: {:?}", res);
    let f_res = res[0].iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();

    let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_res]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}

// 0    300   700   1000      sum(inp)

// 1    1    4     5
// 2    1    16    17
// 3    5    00    5
// 4    17   00    17

// 5    2    5     7
// 6    4    25    29          emb2
// 7    7    00    7           sum(emb)
// 8    29   00    29          sum(emb2)

// 9    3    6     9
// 10   9    36    45
// 11   9    00    9
// 12   45   00    45
// 13   sum(emb)^2 / sf

// 14   5    7    9             sum(emb)^2
// 15   25   49   81
// 16   sum(emb^2) / sf

// 17   25   49   81
// 18   17   29   45             a - b
// 19   8    20   36

// 20   8    20   36    64        sum(a-b)

// 21   64    2
// 22   0                         sum(a-b) / 2
// 23   32

// 24   32
// 25   1000                      add
// 26   1032
