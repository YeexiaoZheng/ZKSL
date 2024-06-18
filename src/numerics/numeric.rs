use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region},
    halo2curves::group::ff::PrimeField,
    plonk::{Advice, Column, Error, Fixed, Selector, TableColumn},
};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum NumericType {
    Dot,
    Accumulator,
}

#[derive(Clone, Debug, Default)]
pub struct _NumericConfig {
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

#[derive(Clone, Debug, Default)]
pub struct NumericConfig {
    pub used_numerics: Arc<BTreeSet<NumericType>>,

    // params
    pub k: usize,
    pub scale_factor: u64,
    pub num_rows: usize,
    pub num_cols: usize,

    // columns
    pub columns: Vec<Column<Advice>>,
    pub constants: Vec<Column<Fixed>>,

    // selectors
    pub use_selectors: bool,
    pub selectors: HashMap<NumericType, Vec<Selector>>,

    // lookup tables
    // pub tables: HashMap<NumericType, Vec<TableColumn>>,
    // pub maps: HashMap<NumericType, Vec<HashMap<i64, i64>>>,
}

pub trait Numeric<F: PrimeField> {
    fn name(&self) -> String;

    fn num_cols_per_op(&self) -> usize;

    fn num_input_cols_per_row(&self) -> usize;

    fn num_output_cols_per_row(&self) -> usize {
        1
    }

    fn op_row_region(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error>;

    // The caller is required to ensure that the inputs are of the correct length.
    fn op_aligned_rows(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Sanity check inputs
        for inp in inputs.iter() {
            assert_eq!(inp.len() % self.num_input_cols_per_row(), 0);
        }

        let outputs = layouter.assign_region(
            || format!("gadget {}", self.name()),
            |mut region| {
                let mut outputs = vec![];
                for i in 0..inputs[0].len() / self.num_input_cols_per_row() {
                    let mut vec_inputs_row = vec![];
                    for inp in inputs.iter() {
                        vec_inputs_row.push(
                            inp[i * self.num_input_cols_per_row()
                                ..(i + 1) * self.num_input_cols_per_row()]
                                .to_vec(),
                        );
                    }
                    let row_outputs =
                        self.op_row_region(&mut region, i, &vec_inputs_row, &constants).unwrap();
                    assert_eq!(row_outputs.len(), 1);
                    outputs.extend(row_outputs);
                }
                Ok(outputs)
            },
        )?;

        Ok(outputs)
    }

    fn forward(
        &self,
        layouter: impl Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error>;
}
