use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region},
    halo2curves::group::ff::PrimeField,
    plonk::{Advice, Column, Error, Fixed, Selector, TableColumn},
};

use crate::utils::math::Int;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum NumericType {
    RowLookUp,
    FieldLookUp,

    Max,
    Add,
    Sub,
    Mul,
    Div,
    Dot,
    Accumulator,
    Relu,
    Exp,
    Ln,
}

#[derive(Clone, Debug, Default)]
pub struct _NumericConfig {
    pub used_numerics: Arc<BTreeSet<NumericType>>,
    pub columns: Vec<Column<Advice>>,
    pub fixed_columns: Vec<Column<Fixed>>,
    pub selectors: HashMap<NumericType, Vec<Selector>>,
    pub tables: HashMap<NumericType, Vec<TableColumn>>,
    pub maps: HashMap<NumericType, Vec<HashMap<Int, Int>>>,
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
    pub min_val: Int,
    pub max_val: Int,
    pub shift_min_val: Int,

    // columns
    pub columns: Vec<Column<Advice>>,
    pub constants: Vec<Column<Fixed>>,

    // lookup tables
    pub tables: HashMap<NumericType, Vec<TableColumn>>,
    pub maps: HashMap<NumericType, Vec<HashMap<Int, Int>>>,

    // selectors
    pub use_selectors: bool,
    pub selectors: HashMap<NumericType, Vec<Selector>>,
}

pub trait Numeric<F: PrimeField> {
    fn name(&self) -> String;

    fn num_cols_per_op(&self) -> usize;

    fn num_input_cols_per_row(&self) -> usize;

    fn num_output_cols_per_row(&self) -> usize {
        1
    }

    // Before use this function, the inputs should be divided into rows, each row is considered as a region.
    // This function will be overridden by the specific numeric operation.
    fn compute_row(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error>;

    // This function is required to ensure that the inputs are of the correct length.
    // The inputs are assumed to be divided into integer rows, with each row having the correct number of columns.
    // The compute_row used in this function is expected to return a single output.
    fn compute_rows(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check that the inputs are of the correct length.
        let cols_per_row = self.num_input_cols_per_row();
        for inp in inputs.iter() {
            assert_eq!(inp.len() % cols_per_row, 0);
        }

        // Process the inputs row by row.
        Ok(layouter.assign_region(
            || format!("numeric {} aligned rows", self.name()),
            |mut region| {
                let mut outputs = vec![];
                for i in 0..inputs[0].len() / cols_per_row {
                    let row_inputs = inputs
                        .iter()
                        .map(|x| x[i * cols_per_row..(i + 1) * cols_per_row].to_vec())
                        .collect::<Vec<_>>();
                    let row_outputs =
                        match self.compute_row(&mut region, i, &row_inputs, &constants) {
                            Ok(res) => res,
                            Err(e) => {
                                panic!("Error in {} numeric op_row_region: {:?}", self.name(), e)
                            }
                        };
                    // Check that the outputs' len is correct.
                    assert_eq!(row_outputs.len(), self.num_output_cols_per_row());
                    outputs.extend(row_outputs);
                }
                Ok(outputs)
            },
        )?)
    }

    // Forward pass for the numeric operation.
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        if inputs[0].len() % self.num_input_cols_per_row() != 0 {
            panic!("Input columns in {} chip are not aligned, please override the forward function to handle this case.", self.name());
        }
        self.compute_rows(
            layouter.namespace(|| format!("{} forward", self.name())),
            inputs,
            constants,
        )
    }
}

pub trait NumericConsumer {
    fn used_numerics(&self) -> Vec<NumericType>;
}
