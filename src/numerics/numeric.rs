use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region},
    halo2curves::group::ff::PrimeField,
    plonk::{Advice, Column, Error, Fixed, Selector, TableColumn},
};
use num_bigint::{BigUint, ToBigUint};
//   use num_traits::cast::ToPrimitive;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum NumericType {
    Dot,
}

#[derive(Clone, Debug, Default)]
pub struct NumericConfig {
    pub used_numerics: Arc<BTreeSet<NumericType>>,
    pub columns: Vec<Column<Advice>>,
    pub fixed_columns: Vec<Column<Fixed>>,
    pub selectors: HashMap<NumericType, Vec<Selector>>,
    pub tables: HashMap<NumericType, Vec<TableColumn>>,
    pub maps: HashMap<NumericType, Vec<HashMap<i64, i64>>>,
    pub scale_factor: u64,
    pub shift_min_val: i64, // MUST be divisible by 2 * scale_factor
    pub num_rows: usize,
    pub num_cols: usize,
    pub k: usize,
    pub eta: f64,
    pub div_outp_min_val: i64,
    pub use_selectors: bool,
    pub num_bits_per_elem: i64,
}

pub trait Numeric<F: PrimeField> {
    fn name(&self) -> String;

    fn forward(&self) -> Result<(), Error> {
        Ok(())
    }
}
