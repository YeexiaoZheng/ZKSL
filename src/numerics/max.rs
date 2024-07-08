use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};

use crate::utils::{
    helpers::{to_field, to_primitive},
    math::max,
};

use super::numeric::{Numeric, NumericConfig, NumericType};

type MaxConfig = NumericConfig;

pub struct MaxChip<F: PrimeField> {
    pub config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> MaxChip<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn num_cols_per_op() -> usize {
        3
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, numeric_config: NumericConfig) -> MaxConfig {
        let selector = meta.complex_selector();
        let columns = &numeric_config.columns;

        meta.create_gate("max gate", |meta| {
            let s = meta.query_selector(selector);
            (0..columns.len() / Self::num_cols_per_op())
                .into_iter()
                .map(|i| {
                    let offset = i * Self::num_cols_per_op();
                    let inp1 = meta.query_advice(columns[offset + 0], Rotation::cur());
                    let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
                    let outp = meta.query_advice(columns[offset + 2], Rotation::cur());
                    s.clone() * (inp1 - outp.clone()) * (inp2 - outp)
                })
                .collect::<Vec<_>>()
        });

        // Need to judge the max value - inp1 or inp2 > 0
        let inp_lookup = numeric_config.tables.get(&NumericType::RowLookUp).unwrap()[0];
        for i in 0..columns.len() / Self::num_cols_per_op() {
            meta.lookup("max inp1", |meta| {
                let s = meta.query_selector(selector);
                let offset = i * Self::num_cols_per_op();
                let inp1 = meta.query_advice(columns[offset + 0], Rotation::cur());
                let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

                vec![(s * (outp - inp1), inp_lookup)]
            });
            meta.lookup("max inp2", |meta| {
                let s = meta.query_selector(selector);
                let offset = i * Self::num_cols_per_op();
                let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
                let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

                vec![(s * (outp - inp2), inp_lookup)]
            });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Max, vec![selector]);

        MaxConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> Numeric<F> for MaxChip<F> {
    fn name(&self) -> String {
        "max".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        Self::num_cols_per_op()
    }

    fn num_input_cols_per_row(&self) -> usize {
        self.config.columns.len() / self.num_cols_per_op() * 2
    }

    fn num_output_cols_per_row(&self) -> usize {
        self.config.columns.len() / self.num_cols_per_op()
    }

    fn compute_row(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        _constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check input shapes
        assert_eq!(inputs.len(), 1);
        let input = &inputs[0];

        // Enable selectors
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Max).unwrap()[0];
            selector.enable(region, row_offset)?;
        }

        // Assign columns
        let mut res = vec![];
        for (i, idx) in (0..input.len()).step_by(2).enumerate() {
            let offset = i * self.num_cols_per_op();
            let in1 = input[idx].copy_advice(
                || "",
                region,
                self.config.columns[offset + 0],
                row_offset,
            )?;
            let in2 = input[idx + 1].copy_advice(
                || "",
                region,
                self.config.columns[offset + 1],
                row_offset,
            )?;
            let max = in1
                .value()
                .zip(in2.value())
                .map(|(x, y)| to_field::<F>(max(to_primitive::<F>(x), to_primitive::<F>(y))));
            res.push(region.assign_advice(
                || "",
                self.config.columns[offset + 2],
                row_offset,
                || max,
            )?);
        }

        Ok(res)
    }

    fn forward(
        &self,
        mut layouter: impl halo2_proofs::circuit::Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Only one input is expected
        assert_eq!(inputs.len(), 1);
        let mut input = inputs[0].clone();
        let first = input[0];

        // Pad the input to the correct length by filling with the first element
        let cols_per_row = self.num_input_cols_per_row();
        while input.len() % cols_per_row != 0 {
            input.push(first);
        }

        // Get initial outputs
        let mut outputs = self.compute_rows(
            layouter.namespace(|| "max forward"),
            &vec![input],
            constants,
        )?;

        // Compute log2^n times until there is only one output
        let times = (outputs.len() as f64).log2().ceil() as usize;
        for _ in 0..times {
            while outputs.len() % cols_per_row != 0 {
                outputs.push(first.clone());
            }
            outputs = self.compute_rows(
                layouter.namespace(|| "max forward"),
                &vec![outputs.iter().collect::<Vec<_>>()],
                constants,
            )?;
        }

        Ok(vec![outputs[0].clone()])
    }
}
