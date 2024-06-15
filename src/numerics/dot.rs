use std::marker::PhantomData;

use halo2_proofs::{
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Expression},
};

use super::numeric::{Numeric, NumericConfig, NumericType};

type DotConfig = NumericConfig;

pub struct DotChip<F: PrimeField> {
    config: DotConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DotChip<F> {
    pub fn construct(config: DotConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> NumericConfig {
        let selector = meta.selector();
        let _columns = &numeric_config.columns;

        meta.create_gate("dot gate", |_meta| {
            let mut _constraints: Vec<Expression<F>> = vec![];
            // TODO:
            _constraints
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Dot, vec![selector]);

        NumericConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> Numeric<F> for DotChip<F> {
    fn name(&self) -> String {
        "Dot".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        self.config.columns.len()
    }
}
