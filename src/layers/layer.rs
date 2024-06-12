use halo2_proofs::{circuit::AssignedCell, halo2curves::ff::PrimeField};
use ndarray::{Array, IxDyn};

use crate::numerics::numeric::NumericType;

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum LayerType {
    FullyConnected,
    ReLU,
    #[default]
    None,
}

pub type CellRc<F> = AssignedCell<F, F>;
pub type AssignedTensor<F> = Array<CellRc<F>, IxDyn>;

#[derive(Clone, Debug, Default)]
pub struct LayerConfig {
    pub layer_type: LayerType,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub layer_params: Vec<i64>, // This is turned into layer specific configurations at runtime
    pub mask: Vec<i64>,
}

pub trait Layer<F: PrimeField> {}

pub trait NumericConsumer {
    fn used_numerics(&self, layer_params: Vec<i64>) -> Vec<NumericType>;
}
