use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error, Expression},
    poly::Rotation,
};

use crate::utils::{
    helpers::{to_field, to_primitive},
    math::{fdiv, Int},
};

use super::{NumericConfig, NumericLayout, NumericType};

type UpdateConfig = NumericConfig;

const COLS_PER_UNIT: usize = 3;

pub struct UpdateLayouter<F: PrimeField> {
    pub config: Rc<UpdateConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> UpdateLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> UpdateConfig {
        let selector = meta.complex_selector();
        let columns = &numeric_config.columns;
        let one = Expression::Constant(F::ONE);

        // Update gate new_weight = weight - grad / reciprocal_lr  c = a - b / d
        /* 20 - 10 / 5 = 18
           advice                                   |  selector
           ----------------------------------------------------
           weight   grad   new_weight  reciprocal_lr|    1
              a       b         c           d       |    s

           e.g.
           len_col = 5
           20  10   18   00  5  |  1
        */
        // c = a - b / d =====> a - c = b / d
        let natural_lookup = numeric_config
            .tables
            .get(&NumericType::NaturalLookUp)
            .unwrap()[0];
        for i in 0..(columns.len() - 1) / COLS_PER_UNIT {
            let offset = i * COLS_PER_UNIT;
            // Check a - c = b / d =====> b >= (a - c) * d =====> b - (a - c) * d >= 0
            meta.lookup("update lookup 1", |meta| {
                let s = meta.query_selector(selector);
                let a = meta.query_advice(columns[offset], Rotation::cur());
                let b = meta.query_advice(columns[offset + 1], Rotation::cur());
                let c = meta.query_advice(columns[offset + 2], Rotation::cur());
                let d = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

                vec![(s * (b - (a - c) * d), natural_lookup)]
            });
            // Check a - c = b / d =====> b < (a - c + 1) * d =====> (a - c + 1) * d - b - 1 >= 0
            meta.lookup("update lookup 2", |meta| {
                let s = meta.query_selector(selector);
                let a = meta.query_advice(columns[offset], Rotation::cur());
                let b = meta.query_advice(columns[offset + 1], Rotation::cur());
                let c = meta.query_advice(columns[offset + 2], Rotation::cur());
                let d = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

                vec![(
                    s * ((a - c + one.clone()) * d - b - one.clone()),
                    natural_lookup,
                )]
            });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Update, selector);

        UpdateConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for UpdateLayouter<F> {
    fn name(&self) -> String {
        "Update".to_string()
    }

    fn num_rows_per_unit(&self) -> usize {
        1
    }

    fn num_cols_per_row(&self) -> usize {
        // Check it equals to the number in the configure function
        (self.config.columns.len() - 1) / COLS_PER_UNIT
    }

    fn layout_unit(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        _constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check shapes
        let weight = inputs[0].clone();
        let grad = inputs[1].clone();
        assert_eq!(weight.len(), grad.len());
        let columns = &self.config.columns.clone();
        let rlr = self.config.reciprocal_learning_rate as Int;

        let units = self.num_cols_per_row();
        let weight_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT])
            .collect::<Vec<_>>();
        let grad_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT + 1])
            .collect::<Vec<_>>();
        let new_weight_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT + 2])
            .collect::<Vec<_>>();
        // Assign inputs
        let weight = self.assign_row(
            region,
            &weight_cols,
            copy_advice,
            row_offset,
            &weight,
            Some(Self::ZERO),
        )?;
        let grad = self.assign_row(
            region,
            &grad_cols,
            copy_advice,
            row_offset,
            &grad,
            Some(Self::ZERO),
        )?;

        // Assign reciprocal learning rate
        region
            .assign_advice(
                || "",
                columns[columns.len() - 1],
                row_offset,
                || Value::known(F::from(rlr as u64)),
            )
            .unwrap();

        // Enable the selector
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Update).unwrap();
            selector.enable(region, row_offset).unwrap();
        }

        Ok(weight
            .iter()
            .zip(grad.iter())
            .enumerate()
            .map(|(idx, (a, b))| {
                // Calculate value of (a - b / rlr)
                let res = a
                    .value()
                    .zip(b.value())
                    .map(|(a, b)| to_field(to_primitive(a) - fdiv(to_primitive(b), rlr)));
                region
                    .assign_advice(|| "", new_weight_cols[idx], row_offset, || res)
                    .unwrap()
            })
            .collect::<Vec<_>>())
    }

    fn layout_customise(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        rows_per_unit: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<(Vec<AssignedCell<F, F>>, usize), Error> {
        // Check shapes
        let input1 = inputs[0].clone();
        let input2 = inputs[1].clone();
        assert_eq!(input1.len(), input2.len());
        let original_input_len = input1.len();
        let cols_per_row = self.num_cols_per_row();

        // Calculate the number of units needed
        let used_units = self.used_units(input1.len());

        let output = input1
            .chunks(cols_per_row)
            .zip(input2.chunks(cols_per_row).enumerate())
            .map(|(inp1, (i, inp2))| {
                self.layout_unit(
                    region,
                    row_offset + i * rows_per_unit,
                    copy_advice,
                    &vec![inp1.to_vec(), inp2.to_vec()],
                    constants,
                )
                .unwrap()
            })
            .flatten()
            .collect::<Vec<_>>();

        let row_offset = row_offset + used_units * rows_per_unit;
        Ok((output[0..original_input_len].to_vec(), row_offset))
    }
}
