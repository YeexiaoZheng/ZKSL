use std::marker::PhantomData;

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::ShapeError;

use crate::{
    numerics::numeric::{Numeric, NumericConfig, NumericType},
    utils::helpers::{AssignedTensor, AssignedTensorRef, Tensor},
};

use super::layer::{ConfigLayer, Layer, LayerConfig, NumericConsumer};

#[derive(Clone, Debug, Default)]
pub struct FullyConnectedLayer<F: PrimeField> {
    pub config: LayerConfig<F>,
}

impl<F: PrimeField> FullyConnectedLayer<F> {
    pub fn construct(config: LayerConfig<F>) -> Self {
        Self { config }
    }
}

impl<F: PrimeField> ConfigLayer<F> for FullyConnectedLayer<F> {
    fn config(&self) -> &LayerConfig<F> {
        &self.config
    }

    fn forward(&self, input: Tensor) -> Result<Tensor, ShapeError> {
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
}

#[derive(Clone, Debug, Default)]
pub struct FullyConnectedChip<F: PrimeField> {
    pub config: LayerConfig<F>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> Layer<F> for FullyConnectedChip<F> {
    fn _forward(&self, input: Tensor) -> Result<Tensor, ShapeError> {
        Ok(FullyConnectedLayer::construct(self.config.clone()).forward(input)?)
    }

    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        input: AssignedTensorRef<F>,
    ) -> Result<AssignedTensor<F>, ShapeError> {
        let weight = self.config.f_weight.clone();
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
