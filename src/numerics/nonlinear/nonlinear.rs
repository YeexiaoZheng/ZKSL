use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error, Selector},
    poly::Rotation,
};

use crate::{
    numerics::numeric::{Numeric, NumericConfig, NumericType},
    utils::helpers::convert_to_u128,
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
            None => panic!("Input lookup table not found"),
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

        let table_col = config.tables.get(&numeric_type).unwrap()[1];

        let shift_pos_i64 = -config.shift_min_val;
        let shift_pos = F::from(shift_pos_i64 as u64);
        layouter.assign_table(
            || "non linear table",
            |mut table| {
                for i in 0..config.num_rows {
                    let i = i as i64;
                    // FIXME: refactor this
                    let tmp = self.get_val_in_map(i);
                    let val = if i == 0 {
                        F::ZERO
                    } else {
                        if tmp >= 0 {
                            F::from(tmp as u64)
                        } else {
                            let tmp = tmp + shift_pos_i64;
                            F::from(tmp as u64) - shift_pos
                        }
                    };
                    table.assign_cell(
                        || "non linear cell",
                        table_col,
                        i as usize,
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
        let shift_val_pos_i64 = -numeric_config.shift_min_val;
        let shift_val_pos = F::from(shift_val_pos_i64 as u64);
        let min_val = numeric_config.min_val;

        if numeric_config.use_selectors {
            let selector = self.get_selector();
            selector.enable(region, row_offset)?;
        }

        let mut outps = vec![];
        for i in 0..input.len() {
            let offset = i * 2;
            input[i].copy_advice(|| "", region, columns[offset + 0], row_offset)?;
            let outp = input[i].value().map(|x: &F| {
                let pos =
                    convert_to_u128(&(*x + shift_val_pos)) as i128 - shift_val_pos_i64 as i128;
                let x = pos as i64 - min_val;
                let val = self.get_val_in_map(x);
                if x == 0 {
                    F::ZERO
                } else {
                    if val >= 0 {
                        F::from(val as u64)
                    } else {
                        let val_pos = val + shift_val_pos_i64;
                        F::from(val_pos as u64) - F::from(shift_val_pos_i64 as u64)
                    }
                }
            });

            let outp = region.assign_advice(
                || "nonlinearity",
                columns[offset + 1],
                row_offset,
                || outp,
            )?;
            outps.push(outp);
        }

        Ok(outps)
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
