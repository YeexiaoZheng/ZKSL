// layer mods
pub mod concat;
pub mod gemm;
pub mod relu;
pub mod softmax;

// default none layer
pub mod none;

// operation types and traits are defined here
use std::{collections::HashMap, hash::Hash};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::ShapeError;

use crate::utils::{
    helpers::{AssignedTensor, AssignedTensorRef, CellRc},
    math::Int,
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
        constants: &HashMap<Int, CellRc<F>>,
        attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError>;

    fn backward(
        &self,
        layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &HashMap<Int, CellRc<F>>,
        attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError>;
}
