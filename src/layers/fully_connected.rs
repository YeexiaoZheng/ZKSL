use std::marker::PhantomData;

use halo2_proofs::halo2curves::ff::PrimeField;
use ndarray::ShapeError;

use crate::numerics::numeric::NumericType;

use super::layer::{Layer, LayerConfig, NumericConsumer, Tensor};

#[derive(Clone, Debug, Default)]
pub struct FullyConnectedConfig {
    pub normalize: bool, // Should be true
}

impl FullyConnectedConfig {
    pub fn construct(normalize: bool) -> Self {
        Self { normalize }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FullyConnectedChip<F: PrimeField> {
    pub config: LayerConfig<F>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> Layer<F> for FullyConnectedChip<F> {
    fn _forward(&self, input: Tensor) -> Result<Tensor, ShapeError> {
        assert_eq!(input.ndim(), 2);
        assert_eq!(input.ndim(), self.config.input_shape.len());
        assert_eq!(self.config.input_shape[1], self.config.weight_shape[0]);
        assert_eq!(self.config.weight_shape[1], self.config.output_shape[0]);
        
        let input_shape = (self.config.input_shape[0], self.config.input_shape[1]);
        let input = input.into_shape(input_shape)?;
        let weight_shape = (self.config.weight_shape[0], self.config.weight_shape[1]);
        let weight = self.config.o_weight.clone().into_shape(weight_shape)?;

        Ok(input.dot(&weight).into_dyn())
    }

    fn forward(&self) {
        todo!()
    }
    
}

impl<F: PrimeField> NumericConsumer for FullyConnectedChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        let mut numerics = vec![];
        numerics.push(NumericType::Dot);
        numerics
    }
}
