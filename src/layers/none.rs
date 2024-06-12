use halo2_proofs::halo2curves::ff::PrimeField;

use crate::numerics::numeric::NumericType;

use super::layer::{Layer, NumericConsumer};

pub struct NoneChip {}

impl<F: PrimeField> Layer<F> for NoneChip {
    
}

impl NumericConsumer for NoneChip {
    fn used_numerics(&self, _layer_params: Vec<i64>) -> Vec<NumericType> {
        vec![]
    }
}