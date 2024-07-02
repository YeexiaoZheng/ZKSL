use std::{collections::HashMap, hash::Hash};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::ShapeError;

use crate::{
    numerics::numeric::NumericType,
    utils::helpers::{AssignedTensor, AssignedTensorRef, CellRc},
};

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum OPType {
    GEMM,
    Conv,
    ReLU,
    SoftMax,
    Constant,
    Reshape,
    Concat,
    #[default]
    None,
}

pub trait Operation<F: PrimeField> {
    fn forward(
        &self,
        layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &HashMap<i64, CellRc<F>>,
        attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError>;

    fn backward(
        &self,
        layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &HashMap<i64, CellRc<F>>,
        attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError>;
}
