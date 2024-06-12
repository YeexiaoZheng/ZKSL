use std::marker::PhantomData;

use halo2_proofs::{halo2curves::ff::PrimeField, plonk::ConstraintSystem};

use super::numeric::{Numeric, NumericConfig};

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
        mumeric_config: NumericConfig,
    ) -> NumericConfig {
        NumericConfig { ..mumeric_config }
    }
}

impl<F: PrimeField> Numeric<F> for DotChip<F> {
    fn typename(&self) -> String {
        "Dot".to_string()
    }
}
