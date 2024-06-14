use halo2_proofs::halo2curves::ff::PrimeField;
use ndarray::ShapeError;

use crate::numerics::numeric::NumericType;

use super::layer::{Layer, NumericConsumer, Tensor};

#[derive(Clone, Debug, Default)]
pub struct NoneChip {}

impl<F: PrimeField> Layer<F> for NoneChip {
    fn _forward(&self, _input: Tensor) -> Result<Tensor, ShapeError> {
        todo!()
    }

    fn forward(&self) {
        todo!()
    }
}

impl NumericConsumer for NoneChip {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![]
    }
}
