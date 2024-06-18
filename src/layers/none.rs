use std::collections::HashMap;

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::ShapeError;

use crate::{numerics::numeric::NumericType, utils::helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor}};

use super::layer::{ConfigLayer, Layer, LayerConfig, NumericConsumer};

#[derive(Clone, Debug, Default)]
pub struct NoneLayer<F: PrimeField> {
    pub config: LayerConfig<F>,
}

impl<F: PrimeField> NoneLayer<F> {
    pub fn construct(config: LayerConfig<F>) -> Self {
        Self { config }
    }
}

impl<F: PrimeField> ConfigLayer<F> for NoneLayer<F> {
    fn config(&self) -> &LayerConfig<F> {
        &self.config
    }

    fn forward(&self, input: Vec<Tensor>) -> Result<Tensor, ShapeError> {
        Ok(input[0].clone())
    }
}

#[derive(Clone, Debug, Default)]
pub struct NoneChip {}

impl<F: PrimeField> Layer<F> for NoneChip {
    fn _forward(&self, _input: Tensor) -> Result<Tensor, ShapeError> {
        todo!()
    }
    
    fn forward(
        &self,
        _layouter: impl Layouter<F>,
        _inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &HashMap<i64, CellRc<F>>,
        _attributes: &HashMap<String, f64>
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        todo!()
    }
}

impl NumericConsumer for NoneChip {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![]
    }
}
