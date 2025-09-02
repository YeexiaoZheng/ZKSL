use halo2_proofs::dev::MockProver;
use halo2_proofs::halo2curves::bn256::Fr;
use ndarray::Array;
use std::collections::BTreeMap;
use zkdeepfm::circuits::max_pool_circuit::MaxPoolCircuit;
use zkdeepfm::numeric::NumericConfig;
use zkdeepfm::utils::helpers::{
    configure_static, configure_static_numeric_config_default, set_log_level, to_field,
};
use zkdeepfm::utils::math::Int;

fn main() {
    // Logging
    set_log_level(log::LevelFilter::Debug);

    type F = Fr;

    let strides: Vec<f64> = vec![1.0, 1.0];
    let kernel_shape: Vec<f64> = vec![2.0, 2.0];
    let mut b_tree_map = BTreeMap::new();
    b_tree_map.insert("strides".to_string(), strides);
    b_tree_map.insert("kernel_shape".to_string(), kernel_shape);

    // let numeric_config = configure_static_numeric_config_default();
    let k = 15;
    let num_cols = 4;
    let scale_factor = 1;
    // let stride = (1, 1);
    // let padding = PaddingEnum::Valid;
    // let pool_size = (2, 2);
    // let pool_stride = (1, 1);

    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        batch_size: 1,
        use_selectors: true,
        ..configure_static_numeric_config_default()
    });

    // input_shape: batch * C_in * H * W
    // let input = vec![ [ 1, 2, 3 ]
    //                   [ 4, 5, 6 ]
    //                   [ 7, 8, 9 ] ];
    let input = vec![10, 2, 30, 4, 5, 6, 70, 8, 90];
    let input_shape = vec![1, 1, 3, 3];
    // let input = vec![10,2,30, 4,5,6, 70,8,90, 100,11,120, 13,14,15, 160,17,180];
    // let input_shape = vec![1, 2, 3, 3];
    let input = input
        .iter()
        .map(|x| x * numeric_config.scale_factor as Int)
        .collect::<Vec<_>>();
    let input = Array::from_shape_vec(input_shape.clone(), input)
        .unwrap()
        .into_dyn();
    println!("test_max_pool, input:\n {:?}", input);

    // grad_shape: batch * C_out * oh * ow
    // let inpgrad = vec![ [ 1, 2 ]
    //                     [ 3, 4 ] ];
    let inpgrad = vec![1, 2, 3, 4];
    let inpgrad_shape = vec![1, 1, 2, 2];
    // let inpgrad = vec![1,2,3,4,5,6,7,8];
    // let inpgrad_shape = vec![1, 2, 2, 2];
    let inpgrad = inpgrad
        .iter()
        .map(|x| x * numeric_config.scale_factor as Int)
        .collect::<Vec<_>>();
    let inpgrad = Array::from_shape_vec(inpgrad_shape.clone(), inpgrad)
        .unwrap()
        .into_dyn();
    println!("test_max_pool, inpgrad:\n {:?}", inpgrad);

    let f_input = input.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_inpgrad = inpgrad
        .iter()
        .map(|x| to_field::<F>(*x))
        .collect::<Vec<_>>();
    let f_input = Array::from_shape_vec(input_shape, f_input)
        .unwrap()
        .into_dyn();
    let f_inpgrad = Array::from_shape_vec(inpgrad_shape, f_inpgrad)
        .unwrap()
        .into_dyn();

    let circuit = MaxPoolCircuit::construct(f_input, f_inpgrad);

    // forward
    // let res = circuit
    //     .forward(&vec![input.clone()], &numeric_config, &b_tree_map)
    //     .unwrap();
    // // println!("test_max_pool, res: {:?}", res);
    // let f_res = res[0].iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    // let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_res]).unwrap();

    // backward
    let res = circuit
        .backward(&vec![inpgrad, input], &numeric_config, &b_tree_map)
        .unwrap();
    let f_res = res[0].iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_res]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
