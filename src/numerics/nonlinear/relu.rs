use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
};

use crate::{
    numerics::numeric::{Numeric, NumericConfig, NumericType},
    utils::math::{relu, Int},
};

use super::nonlinear::NonLinearNumeric;

pub struct ReluChip<F: PrimeField> {
    pub config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ReluChip<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> NumericConfig {
        Self::_configure(
            meta,
            numeric_config,
            NumericType::Relu,
            NumericType::FieldLookUp,
        )
    }
}

impl<F: PrimeField> NonLinearNumeric<F> for ReluChip<F> {
    fn generate_map(_scale_factor: u64, min_val: Int, max_val: Int) -> HashMap<Int, Int> {
        (min_val..max_val)
            .map(|x| (x, relu(x)))
            .collect::<HashMap<_, _>>()
    }

    fn get_numeric_config(&self) -> Rc<NumericConfig> {
        self.config.clone()
    }

    fn get_numeric_type(&self) -> NumericType {
        NumericType::Relu
    }
}

impl<F: PrimeField> Numeric<F> for ReluChip<F> {
    fn name(&self) -> String {
        "Relu".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        <Self as NonLinearNumeric<F>>::num_cols_per_op()
    }

    fn num_input_cols_per_row(&self) -> usize {
        self.config.columns.len() / self.num_cols_per_op()
    }

    fn num_output_cols_per_row(&self) -> usize {
        self.config.columns.len() / self.num_cols_per_op()
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

    fn compute(
        &self,
        mut layouter: impl halo2_proofs::circuit::Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let zero = constants[0];
        let _one = constants[1];
        let mut input = inputs[0].clone();
        let input_len = input.len();
        while input.len() % self.num_input_cols_per_row() != 0 {
            input.push(zero);
        }
        match self.compute_rows(
            layouter.namespace(|| "Relu forward"),
            &vec![input],
            constants,
        ) {
            Ok(outputs) => Ok(outputs[0..input_len].to_vec()),
            Err(e) => panic!("Error in Relu forward compute rows: {}", e),
        }
    }
}
