use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{concatenate, Axis, ShapeError};

use crate::{
    numeric::{NumericConfig, NumericConsumer, NumericType},
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::Int,
    },
};

use super::Operation;

#[derive(Clone, Debug, Default)]
pub struct ConcatChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ConcatChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    // This function is used for non-circuit forward
    pub fn forward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        // concatenate(Axis(axis), views.as_slice()).unwrap_or(tensors[0].clone())
        // TODO: fix this: Axis(1) is hardcoded
        Ok(concatenate![Axis(1), [inputs[0].clone(), inputs[1].clone()]].to_vec())
    }

    // This function is used for non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        Ok(inputs.clone())
    }
}

impl<F: PrimeField> Operation<F> for ConcatChip<F> {
    fn forward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &HashMap<Int, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // TODO: fix this: Axis(1) is hardcoded
        // TODO: fix this: concatenate! is not in circuit!
        Ok(vec![concatenate![
            Axis(1),
            inputs[0].clone(),
            inputs[1].clone()
        ]])
    }

    fn backward(
        &self,
        _layouter: impl Layouter<F>,
        _inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &HashMap<Int, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        Ok(vec![])
    }
}

impl<F: PrimeField> NumericConsumer for ConcatChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![]
    }
}
