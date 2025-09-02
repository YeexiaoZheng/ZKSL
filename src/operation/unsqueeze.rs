use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Axis, ShapeError};

use crate::operation::Operation;
use crate::{
    numeric::{NumericConfig, NumericConsumer, NumericType},
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::Int,
    },
};

// UnsqueezeChip for dimension expansion
#[derive(Clone, Debug, Default)]
pub struct UnsqueezeChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> UnsqueezeChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    // Non-circuit forward: expand (add dimension)
    pub fn forward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        // TODO: fix this: Axis(1) is hardcoded
        Ok(inputs
            .iter()
            .map(|x| x.clone().insert_axis(Axis(1)))
            .collect())
    }

    // Non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        Ok(inputs
            .iter()
            .map(|x| x.clone().index_axis_move(Axis(1), 0))
            .collect())
    }
}

impl<F: PrimeField> Operation<F> for UnsqueezeChip<F> {
    fn forward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // Placeholder for circuit implementation
        // TODO: fix this: Axis(1) is hardcoded
        Ok(inputs
            .iter()
            .map(|x| x.clone().to_owned().insert_axis(Axis(1)))
            .collect())
    }

    fn backward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        Ok(inputs
            .iter()
            .map(|x| x.clone().to_owned().index_axis_move(Axis(1), 0))
            .collect())
    }
}

impl<F: PrimeField> NumericConsumer for UnsqueezeChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![]
    }
}
