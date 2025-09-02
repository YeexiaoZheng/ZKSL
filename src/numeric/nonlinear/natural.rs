use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
};

use crate::numeric::{NumericConfig, NumericType};

// IMPORTANT: It should always be load!!!

pub struct NaturalLookUp<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> NaturalLookUp<F> {
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
        tables.insert(NumericType::NaturalLookUp, vec![lookup]);

        NumericConfig {
            tables,
            ..numeric_config
        }
    }

    pub fn load_lookups(&self, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let lookup = self.numeric_config.tables[&NumericType::NaturalLookUp][0];
        layouter.assign_table(
            || "row input lookup",
            |mut table| {
                for i in 0..self.numeric_config.num_rows {
                    table.assign_cell(
                        || "mod lookup",
                        lookup,
                        i,
                        || Value::known(F::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }
}
