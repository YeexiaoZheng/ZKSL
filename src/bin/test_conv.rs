use halo2_proofs::dev::MockProver;
use halo2_proofs::halo2curves::bn256::Fr;
use ndarray::Array;
use std::collections::BTreeMap;
use zkdeepfm::circuits::conv_circuit::ConvCircuit;
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
    let pads: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0];
    let mut b_tree_map = BTreeMap::new();
    b_tree_map.insert("strides".to_string(), strides);
    b_tree_map.insert("pads".to_string(), pads);

    // let numeric_config = configure_static_numeric_config_default();
    let k = 15;
    let num_cols = 4;
    let scale_factor = 1;

    let numeric_config = configure_static(NumericConfig {
        k,
        num_cols,
        scale_factor,
        batch_size: 1,
        use_selectors: true,
        ..configure_static_numeric_config_default()
    });

    // input_shape: batch * C_in * H * W
    // let input = vec![ [ 1,  2,  3,  4 ]
    //                   [ 5,  6,  7,  8 ]
    //                   [ 9, 10, 11, 12 ]
    //                   [12, 14, 15, 16 ] ];
    let input = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input_shape = vec![1, 1, 4, 4];
    // let input = vec![1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15, 16,17,18];
    // let input_shape = vec![1, 2, 3, 3];
    let input = input
        .iter()
        .map(|x| (x * numeric_config.scale_factor as f64) as Int)
        .collect::<Vec<_>>();
    let input = Array::from_shape_vec(input_shape.clone(), input)
        .unwrap()
        .into_dyn();
    println!("test_conv, input: {:?}", input);

    // weight_shape: C_out * C_in * K_h * K_w
    // let weight = vec![ [ 1, 1 ]
    //                    [ 1, 1 ] ];
    let weight = vec![1.0, 2.0, 3.0, 4.0];
    let weight_shape = vec![1, 1, 2, 2];
    // let weight = vec![1,2,3,4,1,2,3,4];
    // let weight_shape = vec![1, 2, 2, 2];
    // let weight = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let weight = weight
        .iter()
        .map(|x| (x * numeric_config.scale_factor as f64) as Int)
        .collect::<Vec<_>>();
    let weight = Array::from_shape_vec(weight_shape.clone(), weight)
        .unwrap()
        .into_dyn();
    println!("test_conv, weight: {:?}", weight);

    // bias_shape: C_out
    // let biases = [1]
    let biases = vec![-0.11695034056901932];
    let biases_shape = vec![1];
    let biases = biases
        .iter()
        .map(|x| (x * numeric_config.scale_factor as f64) as Int)
        .collect::<Vec<_>>();
    let biases = Array::from_shape_vec(biases_shape.clone(), biases)
        .unwrap()
        .into_dyn();
    println!("test_conv, biases: {:?}", biases);

    // grad_shape: batch * C_out * oh * ow
    // let inpgrad = vec![ [ 1, 1, 1 ]
    //                     [ 1, 1, 1 ]
    //                     [ 1, 1, 1 ] ];
    let inpgrad = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let inpgrad_shape = vec![1, 1, 3, 3];
    let inpgrad = inpgrad
        .iter()
        .map(|x| (x * numeric_config.scale_factor as f64) as Int)
        .collect::<Vec<_>>();
    let inpgrad = Array::from_shape_vec(inpgrad_shape.clone(), inpgrad)
        .unwrap()
        .into_dyn();
    println!("test_conv, inpgrad: {:?}", inpgrad);

    let f_input = input.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_weight = weight.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_biases = biases.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_inpgrad = inpgrad
        .iter()
        .map(|x| to_field::<F>(*x))
        .collect::<Vec<_>>();
    let f_input = Array::from_shape_vec(input_shape, f_input)
        .unwrap()
        .into_dyn();
    let f_weight = Array::from_shape_vec(weight_shape, f_weight)
        .unwrap()
        .into_dyn();
    let f_biases = Array::from_shape_vec(biases_shape, f_biases)
        .unwrap()
        .into_dyn();
    let f_inpgrad = Array::from_shape_vec(inpgrad_shape, f_inpgrad)
        .unwrap()
        .into_dyn();

    let circuit = ConvCircuit::construct(f_input, f_weight, f_biases, f_inpgrad);

    // forward
    // let res = circuit
    //     .forward(&vec![input.clone(), weight.clone(), biases], &numeric_config, &b_tree_map)
    //     .unwrap();
    // // println!("test_conv, res: {:?}", res);
    // let f_res = res[0].iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    // let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_res]).unwrap();
    // assert_eq!(prover.verify(), Ok(()));

    // backward
    let res = circuit
        .backward(
            &vec![inpgrad, input.clone(), weight.clone()],
            &numeric_config,
            &b_tree_map,
        )
        .unwrap();
    // println!("test_conv, res: {:?}", res);
    let (dx, dw, db) = (&res[0], &res[1], &res[2]);
    // println!("test_conv, dx: {:?}", dx);
    // println!("test_conv, dw: {:?}", dw);
    // println!("test_conv, db: {:?}", db);
    let f_dx = dx.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let _f_dw = dw.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let _f_db = db.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_dx]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}
