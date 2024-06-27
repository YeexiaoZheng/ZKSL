use std::{
    collections::BTreeSet,
    rc::Rc,
    sync::{Arc, Mutex},
};

use halo2_proofs::{circuit::AssignedCell, halo2curves::ff::PrimeField};
use lazy_static::lazy_static;
use ndarray::{Array, ArrayView, IxDyn};
use num_bigint::BigUint;
use num_traits::ToPrimitive;

use crate::numerics::numeric::{NumericConfig, NumericType};

use super::math::Int;

pub type Tensor = Array<i64, IxDyn>;
pub type FieldTensor<F> = Array<F, IxDyn>;
pub type CellRc<F> = Rc<AssignedCell<F, F>>;
pub type AssignedTensor<F> = Array<CellRc<F>, IxDyn>;
pub type AssignedTensorRef<'a, F> = ArrayView<'a, CellRc<F>, IxDyn>;

lazy_static! {
    pub static ref NUMERIC_CONFIG: Mutex<NumericConfig> = Mutex::new(NumericConfig::default());
}

pub fn configure_static_numeric_config(
    k: usize,
    num_cols: usize,
    scale_factor: u64,
    used_numerics: BTreeSet<NumericType>,
) {
    let nconfig = &NUMERIC_CONFIG;
    let cloned = nconfig.lock().unwrap().clone();
    // To ensure that max_val - min_val = num_rows
    *nconfig.lock().unwrap() = NumericConfig {
        k,
        scale_factor,
        num_rows: (1 << k) - 10 + 1,
        num_cols,
        min_val: -(1 << (k - 1)),
        max_val: (1 << (k - 1)) - 10,
        use_selectors: true,
        used_numerics: Arc::new(used_numerics),
        ..cloned
    };
}

pub fn to_field<F: PrimeField>(x: Int) -> F {
    let bias = 1 << 31;
    let x_pos = x + bias;
    F::from(x_pos as u64) - F::from(bias as u64)
}

pub fn to_primitive<F: PrimeField>(x: &F) -> Int {
    let bias = 1 << 31;
    let fbias = F::from(bias as u64);
    let big = BigUint::from_bytes_le(&(*x + fbias).to_repr().as_ref());
    let big = big.to_u128().unwrap();
    big as Int - bias
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
