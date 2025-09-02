// pub loss mods
pub mod mse;
pub mod sigmoid;
pub mod softmax;

// loss types and traits are defined here
use std::collections::BTreeMap;

use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    halo2curves::ff::PrimeField,
};
use ndarray::ShapeError;

use crate::utils::{
    helpers::{AssignedTensor, AssignedTensorRef, CellRc},
    math::Int,
};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub enum LossType {
    MSE,
    #[default]
    SoftMax,
    Sigmoid,
}

pub trait Loss<F: PrimeField> {
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

    // This function only returns the gradient
    fn compute(
        &self,
        layouter: impl Layouter<F>,
        input: &AssignedTensorRef<F>,
        label: &AssignedTensorRef<F>,
        constants: &BTreeMap<Int, CellRc<F>>,
    ) -> Result<AssignedTensor<F>, ShapeError>;
}
