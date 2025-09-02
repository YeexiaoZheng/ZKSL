// numeric mods
pub mod add;
pub mod add_same;
pub mod div;
pub mod div_same;
pub mod div_sf;
pub mod dot_vec;
pub mod max;
pub mod mul;
pub mod mul_same;
pub mod square;
pub mod sub;
pub mod sub_same;
pub mod sum;
pub mod update;

// nonlinear mods by lookup tables
pub mod nonlinear;

// numeric types and traits are defined here
use crate::utils::math::Int;
use halo2_proofs::{
    circuit::{AssignedCell, Region, Value},
    halo2curves::group::ff::PrimeField,
    plonk::{Advice, Column, Error, Fixed, Selector, TableColumn},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum NumericType {
    NaturalLookUp, // !IMPORTANT: dont change this position of enum

    Add,
    AddSame,
    Sub,
    SubSame,
    Mul,
    MulSame,
    Div,
    DivSame,
    Square,
    Sum,

    DivSF,

    DotVec,

    Max,

    Relu,
    Exp,
    Ln,
    Gather,

    Update,
}

#[derive(Clone, Debug, Default)]
pub struct NumericConfig {
    pub used_numerics: BTreeSet<NumericType>,

    /// circuit params
    pub k: u32,
    pub scale_factor: u64,
    pub num_rows: usize,
    pub num_cols: usize,
    pub max_val: Int,
    pub min_val: Int,

    /// assignment columns
    pub assigned: Vec<Column<Advice>>,
    pub assigned_num_cols: usize,

    /// witness columns
    pub columns: Vec<Column<Advice>>,
    pub constants: Vec<Column<Fixed>>,

    /// lookup tables
    pub tables: BTreeMap<NumericType, Vec<TableColumn>>, // { type: [input, output] }
    pub maps: BTreeMap<NumericType, BTreeMap<Int, Int>>,
    /// gather lookup advice columns
    pub gather_lookup: Vec<Column<Advice>>,
    pub gather_selector: Option<Selector>,

    /// selectors
    pub use_selectors: bool,
    pub selectors: BTreeMap<NumericType, Selector>,

    /// random vector size
    pub random_size: usize,

    /// commitment
    pub commitment: bool,

    /// TODO: feature params(need to be deleted)
    pub feature_num: Int,

    /// learning params
    pub batch_size: usize,
    pub reciprocal_learning_rate: Int,
}

pub trait NumericLayout<F: PrimeField> {
    /// Constant value of zero.
    const ZERO: F = F::ZERO;
    /// Constant value of one.
    const ONE: F = F::ONE;
    /// Constant value from a integer.
    fn const_from(val: Int) -> F {
        F::from(val as u64)
    }

    /// Name of the numeric layouter.
    fn name(&self) -> String;

    /// Number of rows per unit.
    fn num_rows_per_unit(&self) -> usize;

    /// Number of columns for input per row used.
    fn num_cols_per_row(&self) -> usize;

    /// Number of unit used.
    fn used_units(&self, len: usize) -> usize {
        (len + self.num_cols_per_row() - 1) / self.num_cols_per_row()
    }

    /// Before use this function, the inputs should be divided into units.
    /// Operation layer can use it directly to layout the units for better assign performance.
    /// This function will be overridden by the specific numeric operation.
    fn layout_unit(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error>;

    /// Layout numerics in default way.
    /// Need to return row_offset.
    fn layout(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<(Vec<AssignedCell<F, F>>, usize), Error> {
        self.layout_customise(
            region,
            row_offset,
            self.num_rows_per_unit(),
            true,
            inputs,
            constants,
        )
    }

    /// Layout numerics in default way.
    /// Need to return row_offset.
    fn layout_customise(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        rows_per_unit: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<(Vec<AssignedCell<F, F>>, usize), Error>;

    /// Assign a row in the region by use correct size of cells.
    fn assign_row(
        &self,
        region: &mut Region<F>,
        columns: &Vec<Column<Advice>>,
        copy_advice: bool,
        row_offset: usize,
        cells: &Vec<&AssignedCell<F, F>>,
        pad: Option<F>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let mut cells = cells.iter().map(|cell| (*cell).clone()).collect::<Vec<_>>();

        if copy_advice {
            cells = cells
                .iter()
                .enumerate()
                .map(|(idx, cell)| {
                    cell.copy_advice(|| "", region, columns[idx], row_offset)
                        .unwrap()
                })
                .collect::<Vec<_>>();
        }

        match pad {
            Some(pad) => {
                if cells.len() < self.num_cols_per_row() {
                    let pads = self.pad_row(
                        region,
                        columns,
                        row_offset,
                        cells.len(),
                        self.num_cols_per_row(),
                        pad,
                    )?;
                    cells.extend(pads.into_iter());
                }
            }
            None => {
                assert_eq!(cells.len(), self.num_cols_per_row());
            }
        }

        Ok(cells)
    }

    /// Pad a row by assign a constant value of "pad" to the cells.
    fn pad_row(
        &self,
        region: &mut Region<F>,
        columns: &Vec<Column<Advice>>,
        row_offset: usize,
        start: usize,
        end: usize,
        pad: F,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        Ok((start..end)
            .map(|idx| {
                region
                    .assign_advice(|| "", columns[idx], row_offset, || Value::known(pad))
                    .unwrap()
            })
            .collect())
    }
}

pub trait NumericConsumer {
    fn used_numerics(&self) -> Vec<NumericType>;
}
