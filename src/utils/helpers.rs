use std::{rc::Rc, sync::Mutex};

use halo2_proofs::circuit::AssignedCell;
use lazy_static::lazy_static;
use ndarray::{Array, ArrayView, IxDyn};

use crate::numerics::numeric::NumericConfig;

pub type Tensor = Array<i64, IxDyn>;
pub type FieldTensor<F> = Array<F, IxDyn>;
pub type CellRc<F> = Rc<AssignedCell<F, F>>;
pub type AssignedTensor<F> = Array<CellRc<F>, IxDyn>;
pub type AssignedTensorRef<'a, F> = ArrayView<'a, AssignedCell<F, F>, IxDyn>;

lazy_static! {
    pub static ref NUMERIC_CONFIG: Mutex<NumericConfig> = Mutex::new(NumericConfig::default());
}
