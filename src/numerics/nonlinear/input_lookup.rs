use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
};

use crate::numerics::numeric::{Numeric, NumericConfig, NumericType};

use super::nonlinear::NonLinearNumeric;

pub struct InputLookupChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> InputLookupChip<F> {
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
        Self::_configure(meta, numeric_config, NumericType::InputLookup)
    }
}

impl<F: PrimeField> NonLinearNumeric<F> for InputLookupChip<F> {
    fn generate_map(scale_factor: u64, min_val: i64, num_rows: i64) -> HashMap<i64, i64> {
        (0..num_rows)
            .map(|i| (i, (i * scale_factor as i64) + min_val))
            .collect::<HashMap<_, _>>()
    }

    fn _configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
        numeric_type: NumericType,
    ) -> NumericConfig {
        assert_eq!(numeric_type, NumericType::InputLookup);
        let lookup = meta.lookup_table_column();
        let mut tables = numeric_config.tables;
        tables.insert(numeric_type, vec![lookup]);

        NumericConfig {
            tables,
            ..numeric_config
        }
    }

    fn load_lookups(&self, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let lookup = self.numeric_config.tables[&NumericType::InputLookup][0];
        layouter.assign_table(
            || "input lookup",
            |mut table| {
                for i in 0..self.numeric_config.num_rows as i64 {
                    table.assign_cell(
                        || "mod lookup",
                        lookup,
                        i as usize,
                        || Value::known(F::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    fn get_numeric_config(&self) -> Rc<NumericConfig> {
        self.numeric_config.clone()
    }

    fn get_numeric_type(&self) -> NumericType {
        NumericType::InputLookup
    }
}

impl<F: PrimeField> Numeric<F> for InputLookupChip<F> {
    fn name(&self) -> String {
        "InputLookup".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        2
    }

    fn num_input_cols_per_row(&self) -> usize {
        1
    }

    fn compute_row(
        &self,
        _region: &mut halo2_proofs::circuit::Region<F>,
        _row_offset: usize,
        _inputs: &Vec<Vec<&halo2_proofs::circuit::AssignedCell<F, F>>>,
        _constants: &Vec<&halo2_proofs::circuit::AssignedCell<F, F>>,
    ) -> Result<Vec<halo2_proofs::circuit::AssignedCell<F, F>>, halo2_proofs::plonk::Error> {
        panic!("InputLookupChip does not support compute_row");
    }
}
