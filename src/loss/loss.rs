use std::collections::HashMap;

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::ShapeError;

use crate::utils::{
    helpers::{AssignedTensor, AssignedTensorRef, CellRc},
    math::Int,
};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum LossType {
    MSE,
    SoftMax,
}

pub trait Loss<F: PrimeField> {
    // This function only returns the gradient
    fn compute(
        &self,
        layouter: impl Layouter<F>,
        input: &AssignedTensorRef<F>,
        label: &Vec<CellRc<F>>,
        constants: &HashMap<Int, CellRc<F>>,
    ) -> Result<AssignedTensor<F>, ShapeError>;
}
