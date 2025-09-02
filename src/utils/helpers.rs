use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Mutex,
};

use halo2_proofs::{
    circuit::{AssignedCell, Value},
    halo2curves::ff::PrimeField,
    plonk::ConstraintSystem,
};
use lazy_static::lazy_static;
use ndarray::{Array, ArrayView, IxDyn};
use num_bigint::BigUint;
use num_traits::ToPrimitive;

use crate::{
    graph::Graph,
    numeric::{NumericConfig, NumericType},
};

use super::math::Int;

pub type Tensor = Array<Int, IxDyn>;
pub type FieldTensor<F> = Array<F, IxDyn>;
pub type ValueTensor<F> = Array<Value<F>, IxDyn>;

pub type CellRc<F> = Rc<AssignedCell<F, F>>;
pub type AssignedTensor<F> = Array<CellRc<F>, IxDyn>;
pub type AssignedTensorRef<'a, F> = ArrayView<'a, CellRc<F>, IxDyn>;

// We can alias Field to support different prime fields
// trait Field = PrimeField;

lazy_static! {
    pub static ref NUMERIC_CONFIG: Mutex<NumericConfig> = Mutex::new(NumericConfig::default());
}

pub fn get_numeric_config() -> NumericConfig {
    NUMERIC_CONFIG.lock().unwrap().clone()
}

pub fn get_circuit_numeric_config<F: PrimeField>(meta: &mut ConstraintSystem<F>) -> NumericConfig {
    let mut numeric_config = NUMERIC_CONFIG.lock().unwrap().clone();
    let lookup_max = (1 << numeric_config.k) - (meta.blinding_factors() + 1) - 1;
    numeric_config.num_rows = lookup_max;
    numeric_config.min_val = -(lookup_max as Int) / 2;
    numeric_config.max_val = lookup_max as Int - (lookup_max as Int) / 2;
    numeric_config
}

pub fn configure_static_numeric_config(
    k: u32,
    num_cols: usize,
    scale_factor: u64,
    batch_size: usize,
    used_numerics: BTreeSet<NumericType>,
) {
    let nconfig = &NUMERIC_CONFIG;
    let cloned = nconfig.lock().unwrap().clone();
    // To ensure that max_val - min_val < num_rows
    *nconfig.lock().unwrap() = NumericConfig {
        k,
        scale_factor,
        num_cols,
        use_selectors: true,
        batch_size,
        used_numerics: used_numerics,
        commitment: true,
        ..cloned
    };
}

pub fn configure_static(numeric_config: NumericConfig) -> NumericConfig {
    let nconfig = &NUMERIC_CONFIG;
    *nconfig.lock().unwrap() = numeric_config.clone();
    numeric_config
}

pub fn configure_static_numeric_config_default() -> NumericConfig {
    let nconfig = &NUMERIC_CONFIG;
    let cloned = nconfig.lock().unwrap().clone();
    let k = 14;
    let new_numeric_config = NumericConfig {
        k,
        scale_factor: 512,
        num_cols: 11,
        use_selectors: true,
        batch_size: 1,
        random_size: 1000,
        assigned_num_cols: 1,
        reciprocal_learning_rate: 1,
        commitment: false,
        ..cloned
    };
    *nconfig.lock().unwrap() = new_numeric_config.clone();
    new_numeric_config
}

pub fn set_log_level(log_level: log::LevelFilter) {
    env_logger::builder().filter_level(log_level).init();
}

pub fn to_field<F: PrimeField>(x: Int) -> F {
    let bias = 1 << 60;
    let x_pos = x + bias;
    F::from(x_pos as u64) - F::from(bias as u64)
}

pub fn to_primitive<F: PrimeField>(x: &F) -> Int {
    let bias = 1 << 60;
    let fbias = F::from(bias as u64);
    let big = BigUint::from_bytes_le(&(*x + fbias).to_repr().as_ref());
    let big = big.to_u128().unwrap();
    big as Int - bias
}

pub fn update_graph(orginal_graph: &Graph, tensor_map: &BTreeMap<String, Tensor>) -> Graph {
    let mut graph = orginal_graph.clone();
    // graph.tensor_map.clear();
    for k in orginal_graph.tensor_map.keys() {
        if tensor_map.contains_key(k) {
            graph
                .tensor_map
                .insert(k.clone(), tensor_map.get(k).unwrap().clone());
        }
    }
    graph
}

// TODO: refactor
pub fn convert_to_u64<F: PrimeField>(x: &F) -> u64 {
    let big = BigUint::from_bytes_le(x.to_repr().as_ref());
    let big_digits = big.to_u64_digits();
    if big_digits.len() > 2 {
        println!("big_digits: {:?}", big_digits);
    }
    if big_digits.len() == 1 {
        big_digits[0] as u64
    } else if big_digits.len() == 0 {
        0
    } else {
        panic!();
    }
}

pub fn convert_to_u128<F: PrimeField>(x: &F) -> u128 {
    let big = BigUint::from_bytes_le(x.to_repr().as_ref());
    big.to_u128().unwrap()
}
