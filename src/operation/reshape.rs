use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::ShapeError;

use crate::{
    numeric::{NumericConfig, NumericConsumer, NumericType},
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::Int,
    },
};

use super::Operation;

// ReshapeChip for reshape vec
// (a,b,c) -> (1, a*b*c)
#[derive(Clone, Debug, Default)]
pub struct ReshapeChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ReshapeChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    // Non-circuit forward: reshape (reduce dimension)
    pub fn forward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        // println!("reshape inputs: {:?}", inputs);
        let input: Tensor = inputs[0].clone();
        let shape = input.shape();
        let new_shape = vec![shape[0], shape[1..].iter().copied().product()];
        let output = input
            .to_shape(ndarray::IxDyn(&new_shape))
            .unwrap()
            .into_dyn()
            .to_owned();
        // println!("reshape non circuit forward: {:?}", output.shape());
        Ok(vec![output])
    }

    // Non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let inpgrad = &inputs[0].clone(); // (1, 4)
        let shape = inputs[1].shape();
        let output = inpgrad.to_shape(shape).unwrap().into_dyn().to_owned();
        // println!("reshape non circuit backward: {:?}", output.shape());
        Ok(vec![output])
    }
}

impl<F: PrimeField> Operation<F> for ReshapeChip<F> {
    fn forward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let input: AssignedTensorRef<F> = inputs[0].clone();
        let shape = input.shape();
        let new_shape = vec![shape[0], shape[1..].iter().copied().product()];
        let output = input
            .to_shape(ndarray::IxDyn(&new_shape))
            .unwrap()
            .into_dyn()
            .to_owned();
        Ok(vec![output])
    }

    fn backward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let inpgrad = &inputs[0].clone(); // (1, 4)
        let shape = inputs[1].shape();
        let output = inpgrad.to_shape(shape).unwrap().into_dyn().to_owned();
        Ok(vec![output])
    }
}

impl<F: PrimeField> NumericConsumer for ReshapeChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![]
    }
}
