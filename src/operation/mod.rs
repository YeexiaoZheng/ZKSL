// layer mods
pub mod add;
pub mod concat;
pub mod fm;
pub mod gather;
pub mod gemm;
pub mod mean;
pub mod relu;
pub mod softmax;
pub mod squeeze;
pub mod unsqueeze;

// default none layer
pub mod conv;
pub mod max_pool;
pub mod none;
pub mod reshape;

// operation types and traits are defined here
use std::{collections::BTreeMap, hash::Hash};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    halo2curves::ff::PrimeField,
};
use ndarray::ShapeError;

use crate::utils::{
    helpers::{AssignedTensor, AssignedTensorRef, CellRc},
    math::Int,
};

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum OPType {
    Concat,
    Gather,
    FM,
    Mean,
    GEMM,
    ReLU,
    SoftMax,
    Sigmoid,
    Unsqueeze,
    Squeeze,
    Add,
    Conv,
    MaxPool,
    Reshape,
    #[default]
    None,
}

pub trait Operation<F: PrimeField> {
    fn get_default_constants<'a>(
        &self,
        constants: &'a BTreeMap<Int, CellRc<F>>,
    ) -> Vec<&'a AssignedCell<F, F>> {
        let zero = constants.get(&0).unwrap();
        let one = constants.get(&1).unwrap();
        vec![zero, one]
    }

    fn get_constants<'a>(
        &self,
        constants: &'a BTreeMap<Int, CellRc<F>>,
        keys: Vec<Int>,
    ) -> Vec<&'a AssignedCell<F, F>> {
        keys.iter()
            .map(|k| constants.get(k).unwrap().as_ref())
            .collect()
    }

    fn forward(
        &self,
        layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        random: &Vec<CellRc<F>>,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError>;

    fn backward(
        &self,
        layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        random: &Vec<CellRc<F>>,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError>;
}
