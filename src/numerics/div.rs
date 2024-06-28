use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};

use crate::utils::helpers::{to_field, to_primitive};

use super::numeric::{Numeric, NumericConfig, NumericType};

type DivConfig = NumericConfig;

// IMPORTANT: Please use div chip every time you need to multiply first input by scale_factor
pub struct DivChip<F: PrimeField> {
    pub config: Rc<DivConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DivChip<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn num_cols_per_op() -> usize {
        4
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, numeric_config: NumericConfig) -> DivConfig {
        let selector = meta.selector();
        let columns = &numeric_config.columns;

        meta.create_gate("div gate", |meta| {
            let s = meta.query_selector(selector);
            (0..columns.len() / Self::num_cols_per_op())
                .into_iter()
                .map(|i| {
                    let offset = i * Self::num_cols_per_op();
                    let inp1 = meta.query_advice(columns[offset + 0], Rotation::cur());
                    let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
                    let outp = meta.query_advice(columns[offset + 2], Rotation::cur());
                    let reminder = meta.query_advice(columns[offset + 3], Rotation::cur());
                    s.clone() * (inp1 - reminder - inp2 * outp)
                })
                .collect::<Vec<_>>()
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Div, vec![selector]);

        DivConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> Numeric<F> for DivChip<F> {
    fn name(&self) -> String {
        "Div".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        Self::num_cols_per_op()
    }

    fn num_input_cols_per_row(&self) -> usize {
        self.config.columns.len() / self.num_cols_per_op()
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
        assert_eq!(inputs.len(), 2);
        let input1 = &inputs[0];
        let input2 = &inputs[1];
        assert_eq!(input1.len(), input2.len());
        assert_eq!(input1.len(), self.num_input_cols_per_row());
        let columns = &self.config.columns;

        // Enable selectors
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Div).unwrap()[0];
            selector.enable(region, row_offset)?;
        }

        // Assign columns
        let mut res = vec![];
        for i in 0..input1.len() {
            let offset = i * self.num_cols_per_op();
            let in1 =
                input1[i].copy_advice(|| "assign in1", region, columns[offset + 0], row_offset)?;
            let in2 =
                input2[i].copy_advice(|| "assign in2", region, columns[offset + 1], row_offset)?;
            let out = in1
                .value()
                .map(|a| a)
                .zip(in2.value().map(|b| b))
                .map(|(a, b)| {
                    let a = to_primitive::<F>(a);
                    let mut b = to_primitive::<F>(b);
                    if b == 0 {
                        println!("Warning!: divide by zero! now change divisor to 1");
                        b = 1;
                    }
                    to_field::<F>(a / b)
                });
            let out =
                region.assign_advice(|| "assign out", columns[offset + 2], row_offset, || out)?;
            let reminder = in1.value().copied() - in2.value().copied() * out.value().copied();
            region.assign_advice(
                || "assign reminder",
                columns[offset + 3],
                row_offset,
                || reminder,
            )?;
            res.push(out);
        }

        Ok(res)
    }

    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check input and weight shapes
        let mut input1 = inputs[0].clone();
        let mut input2 = inputs[1].clone();
        assert_eq!(input1.len(), input2.len());
        let input_len = input1.len();
        let cols_per_row = self.num_input_cols_per_row();
        let one = constants[1];

        // Fill the input and weight columns to be multiple of num_input_cols_per_row
        while input1.len() % cols_per_row != 0 {
            input1.push(&one);
            input2.push(&one);
        }

        // Assign columns row by row
        let res = self.compute_rows(
            layouter.namespace(|| "div rows"),
            &vec![input1, input2],
            constants,
        )?;

        Ok(res[0..input_len].to_vec())
    }
}
