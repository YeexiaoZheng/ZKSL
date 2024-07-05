use std::collections::HashMap;

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};

use crate::utils::helpers::{AssignedTensorRef, CellRc};

pub enum LossType {
    MSE,
    SoftMax,
}

pub trait Loss<F: PrimeField> {
    // returns the loss and gradients
    fn compute(
        &self,
        layouter: impl Layouter<F>,
        inputs: &AssignedTensorRef<F>,
        label: &Vec<CellRc<F>>,
        constants: &HashMap<i64, CellRc<F>>,
    ) -> ();
}
