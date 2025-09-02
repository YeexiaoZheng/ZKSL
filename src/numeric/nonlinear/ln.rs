use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
};

use crate::{
    numeric::{NumericConfig, NumericLayout, NumericType},
    utils::math::{ln, Int},
};

use super::NonLinearNumericLayout;

// IMPORTANT: It returns Ln(x / scale_factor) * scale_factor
pub struct LnLookUp<F: PrimeField> {
    pub config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> LnLookUp<F> {
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
        Self::_configure(meta, numeric_config, NumericType::Ln, false)
    }
}

impl<F: PrimeField> NonLinearNumericLayout<F> for LnLookUp<F> {
    fn generate_map(scale_factor: u64, min_val: Int, max_val: Int) -> BTreeMap<Int, Int> {
        (min_val..max_val)
            .map(|x| {
                let ln = ln(x, scale_factor);
                (x, ln)
            })
            .collect::<BTreeMap<_, _>>()
    }

    fn get_numeric_config(&self) -> Rc<NumericConfig> {
        self.config.clone()
    }

    fn get_numeric_type(&self) -> NumericType {
        NumericType::Ln
    }
}

impl<F: PrimeField> NumericLayout<F> for LnLookUp<F> {
    fn name(&self) -> String {
        "Ln".to_string()
    }

    fn num_rows_per_unit(&self) -> usize {
        <Self as NonLinearNumericLayout<F>>::num_rows_per_unit()
    }

    fn num_cols_per_row(&self) -> usize {
        self.config.columns.len() / 2
    }

    fn layout_unit(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        <Self as NonLinearNumericLayout<F>>::layout_unit(
            &self,
            region,
            row_offset,
            copy_advice,
            inputs,
            constants,
        )
    }

    fn layout_customise(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        rows_per_unit: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<(Vec<AssignedCell<F, F>>, usize), Error> {
        <Self as NonLinearNumericLayout<F>>::layout_customise(
            &self,
            region,
            row_offset,
            rows_per_unit,
            copy_advice,
            inputs,
            constants,
        )
    }
}
