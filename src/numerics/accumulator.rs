use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error, Expression},
    poly::Rotation,
};

use super::numeric::{Numeric, NumericType, NumericConfig};

type AccumulatorConfig = NumericConfig;

pub struct AccumulatorChip<F: PrimeField> {
    pub config: Rc<AccumulatorConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> AccumulatorChip<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> AccumulatorConfig {
        let selector = meta.selector();
        let columns = numeric_config.columns;

        // Create a add gate for the accumulator (accumulator will use it recursively to add all inputs)
        meta.create_gate("accumulator add gate", |meta| {
            let s = meta.query_selector(selector);

            let gate_input = columns[0..columns.len() - 1]
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::cur()))
                .collect::<Vec<_>>();
            let gate_output = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

            let res = gate_input
                .iter()
                .fold(Expression::Constant(F::ZERO), |acc, x| acc + x.clone());

            vec![s * (res - gate_output)]
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Accumulator, vec![selector]);

        AccumulatorConfig {
            columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> Numeric<F> for AccumulatorChip<F> {
    fn name(&self) -> String {
        "Accumulator".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        self.config.columns.len()
    }

    fn num_input_cols_per_row(&self) -> usize {
        self.config.columns.len() - 1
    }

    fn compute_row(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        _constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check input shape
        assert_eq!(inputs.len(), 1);
        let input = &inputs[0];
        let columns = self.config.columns.to_vec();
        assert_eq!(input.len(), columns.len() - 1);

        // Enable the selector
        if self.config.use_selectors {
            let selector = self
                .config
                .selectors
                .get(&NumericType::Accumulator)
                .unwrap()[0];
            selector.enable(region, row_offset)?;
        }

        // Assign the input by copy advice
        input
            .iter()
            .enumerate()
            .map(|(i, cell)| cell.copy_advice(|| "", region, columns[i], row_offset))
            .collect::<Result<Vec<_>, _>>()?;

        // Accumulate the input
        let value = input
            .iter()
            .fold(Value::known(F::ZERO), |acc, x| acc + x.value().copied());

        // Assign the output column
        let res = region.assign_advice(|| "", columns[columns.len() - 1], row_offset, || value)?;

        Ok(vec![res])
    }

    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Only need one input to accumulate and zero constant
        assert_eq!(inputs.len(), 1);
        assert_eq!(constants.len(), 1);
        let cols_per_row = self.num_input_cols_per_row();
        let mut input = inputs[0].clone();
        let zero = constants[0];

        // Pad the input to be a multiple of the number of columns
        while input.len() % cols_per_row != 0 {
            input.push(&zero);
        }
        // Accumulate the input to get initial outputs
        let mut outputs = self.compute_rows(
            layouter.namespace(|| "accumulator forward"),
            &vec![input],
            constants,
        )?;

        // Accumulate recursively until there is only one output
        while outputs.len() != 1 {
            while outputs.len() % cols_per_row != 0 {
                outputs.push(zero.clone());
            }
            outputs = self.compute_rows(
                layouter.namespace(|| "accumulator forward"),
                &vec![outputs.iter().collect::<Vec<_>>()],
                constants,
            )?;
        }

        Ok(outputs)
    }
}
