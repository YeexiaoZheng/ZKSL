use std::{rc::Rc, sync::Mutex};

use halo2_proofs::{
    circuit::AssignedCell,
    halo2curves::ff::PrimeField,
};
use lazy_static::lazy_static;
use ndarray::{Array, ArrayView, IxDyn};

use crate::numerics::numeric::NumericConfig;

pub type Tensor = Array<i64, IxDyn>;
pub type FieldTensor<F> = Array<F, IxDyn>;
pub type CellRc<F> = Rc<AssignedCell<F, F>>;
pub type AssignedTensor<F> = Array<CellRc<F>, IxDyn>;
pub type AssignedTensorRef<'a, F> = ArrayView<'a, CellRc<F>, IxDyn>;

lazy_static! {
    pub static ref NUMERIC_CONFIG: Mutex<NumericConfig> = Mutex::new(NumericConfig::default());
}

pub fn to_field<F: PrimeField>(x: i64) -> F {
    let bias = 1 << 31;
    let x_pos = x + bias;
    F::from(x_pos as u64) - F::from(bias as u64)
}
