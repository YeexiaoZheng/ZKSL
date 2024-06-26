use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
};

use crate::{
    numerics::numeric::{Numeric, NumericConfig, NumericType},
    utils::helpers::to_field,
};

pub struct FieldLookUpChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> FieldLookUpChip<F> {
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
        let lookup = meta.lookup_table_column();
        let mut tables = numeric_config.tables;
        tables.insert(NumericType::FieldLookUp, vec![lookup]);

        NumericConfig {
            tables,
            ..numeric_config
        }
    }

    pub fn load_lookups(&self, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let lookup = self.numeric_config.tables[&NumericType::FieldLookUp][0];
        layouter.assign_table(
            || "field input lookup",
            |mut table| {
                for (offset, x) in
                    (self.numeric_config.min_val..self.numeric_config.max_val).enumerate()
                {
                    table.assign_cell(
                        || "field lookup cell",
                        lookup,
                        offset,
                        || Value::known(to_field::<F>(x)),
                    )?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }
}

impl<F: PrimeField> Numeric<F> for FieldLookUpChip<F> {
    fn name(&self) -> String {
        "FieldLookUp".to_string()
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
        panic!("FieldLookUpChip does not support compute_row");
    }
}
