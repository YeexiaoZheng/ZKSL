use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
};

use crate::{
    numerics::numeric::{Numeric, NumericConfig, NumericType},
    utils::math::ln,
};

use super::nonlinear::NonLinearNumeric;

// IMPORTANT: It returns Ln(x / scale_factor) * scale_factor
pub struct LnChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> LnChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> NumericConfig {
        Self::_configure(meta, numeric_config, NumericType::Relu)
    }
}

impl<F: PrimeField> NonLinearNumeric<F> for LnChip<F> {
    fn generate_map(scale_factor: u64, min_val: i64, num_rows: i64) -> HashMap<i64, i64> {
        (0..num_rows)
            .map(|i| {
                let x = i + min_val;
                let ln = ln(x, scale_factor);
                (x, ln)
            })
            .collect::<HashMap<_, _>>()
    }

    fn get_numeric_config(&self) -> Rc<NumericConfig> {
        self.numeric_config.clone()
    }

    fn get_numeric_type(&self) -> NumericType {
        NumericType::Ln
    }
}

impl<F: PrimeField> Numeric<F> for LnChip<F> {
    fn name(&self) -> String {
        "Ln".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        todo!()
    }

    fn num_input_cols_per_row(&self) -> usize {
        todo!()
    }

    fn compute_row(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        <Self as NonLinearNumeric<F>>::compute_row(&self, region, row_offset, inputs, constants)
    }
}
