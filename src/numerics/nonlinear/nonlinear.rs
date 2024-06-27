use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error, Selector},
    poly::Rotation,
};

use crate::{
    numerics::numeric::{Numeric, NumericConfig, NumericType},
    utils::helpers::{to_field, to_primitive},
};

const NUM_LOOKUP_COLS_PER_OP: usize = 2;

pub trait NonLinearNumeric<F: PrimeField>: Numeric<F> {
    fn generate_map(scale_factor: u64, min_val: i64, num_rows: i64) -> HashMap<i64, i64>;

    fn get_numeric_config(&self) -> Rc<NumericConfig>;

    fn get_numeric_type(&self) -> NumericType;

    fn get_val_in_map(&self, key: i64) -> i64 {
        match self.get_numeric_config().maps.get(&self.get_numeric_type()) {
            Some(map) => map[0].get(&key).unwrap().clone(),
            None => panic!("Map is not found"),
        }
    }

    fn get_selector(&self) -> Selector {
        match self
            .get_numeric_config()
            .selectors
            .get(&self.get_numeric_type())
        {
            Some(selectors) => selectors[0],
            None => panic!("Selector is not found"),
        }
    }

    fn _configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
        numeric_type: NumericType,
    ) -> NumericConfig {
        let s = meta.complex_selector();
        let cols_per_op = NUM_LOOKUP_COLS_PER_OP;
        let columns = numeric_config.columns;

        let mut tables = numeric_config.tables;
        let input_lookup = match tables.get(&NumericType::FieldLookUp) {
            Some(tables) => tables,
            None => panic!("Input field lookup table not found"),
        }[0];
        let output_lookup = meta.lookup_table_column();

        for op_idx in 0..columns.len() / cols_per_op {
            let offset = op_idx * cols_per_op;
            meta.lookup("non-linear lookup", |meta| {
                let s = meta.query_selector(s);
                let input_col = meta.query_advice(columns[offset + 0], Rotation::cur());
                let output_col = meta.query_advice(columns[offset + 1], Rotation::cur());
                vec![
                    (s.clone() * input_col, input_lookup),
                    (s.clone() * output_col, output_lookup),
                ]
            });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(numeric_type, vec![s]);

        tables.insert(numeric_type, vec![input_lookup, output_lookup]);

        let mut maps = numeric_config.maps;
        let non_linear_map = Self::generate_map(
            numeric_config.scale_factor,
            numeric_config.min_val,
            numeric_config.num_rows as i64,
        );
        maps.insert(numeric_type, vec![non_linear_map]);

        NumericConfig {
            columns,
            selectors,
            tables,
            maps,
            ..numeric_config
        }
    }

    fn load_lookups(&self, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let config = self.get_numeric_config();
        let numeric_type = self.get_numeric_type();
        let output_lookup = config.tables.get(&numeric_type).unwrap()[1];

        layouter.assign_table(
            || "non-linear table",
            |mut table| {
                for i in 0..config.num_rows {
                    let x = i as i64 + config.min_val;
                    let val = to_field::<F>(self.get_val_in_map(x));
                    table.assign_cell(
                        || "non-linear cell",
                        output_lookup,
                        i,
                        || Value::known(val),
                    )?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    fn compute_row(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        _constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let numeric_config = self.get_numeric_config();
        let columns = &self.get_numeric_config().columns;
        let input = &inputs[0];

        if numeric_config.use_selectors {
            let selector = self.get_selector();
            selector.enable(region, row_offset)?;
        }

        input
            .iter()
            .enumerate()
            .map(|(i, cell)| cell.copy_advice(|| "", region, columns[i * 2], row_offset))
            .collect::<Result<Vec<_>, _>>()?;

        let res = input
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let value = cell
                    .value()
                    .map(|x| to_field::<F>(self.get_val_in_map(to_primitive::<F>(x))));
                region.assign_advice(|| "non-linear", columns[i * 2 + 1], row_offset, || value)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(res)
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
