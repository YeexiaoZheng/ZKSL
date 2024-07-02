use halo2_proofs::halo2curves::ff::PrimeField;

use crate::numerics::numeric::NumericType;

pub enum LossType {
    MSE,
    SoftMax,
}

pub trait Loss<F: PrimeField> {
    // returns the loss and gradients
    fn compute(
        &self,
        // layouter: impl Layouter<F>,
        // inputs: &Vec<AssignedTensorRef<F>>,
        // constants: &HashMap<i64, CellRc<F>>,
        // attributes: &HashMap<String, f64>,
    ) -> ();
}
