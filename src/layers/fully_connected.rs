use std::marker::PhantomData;

use halo2_proofs::halo2curves::ff::PrimeField;

use crate::numerics::numeric::NumericType;

use super::layer::NumericConsumer;

pub struct FullyConnectedConfig {
    pub normalize: bool, // Should be true
}

impl FullyConnectedConfig {
    pub fn construct(normalize: bool) -> Self {
        Self { normalize }
    }
}

pub struct FullyConnectedChip<F: PrimeField> {
    pub _marker: PhantomData<F>,
    pub config: FullyConnectedConfig,
}

impl<F: PrimeField> NumericConsumer for FullyConnectedChip<F> {
    fn used_numerics(&self, layer_params: Vec<i64>) -> Vec<NumericType> {
        let mut numerics = vec![];
        if self.config.normalize {
            numerics.push(NumericType::Dot);
        }
        numerics
    }
}