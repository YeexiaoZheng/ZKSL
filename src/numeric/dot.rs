use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Column, ConstraintSystem, Error, Expression},
    poly::Rotation,
};

use crate::numeric::accumulator::AccumulatorChip;

use super::{Numeric, NumericConfig, NumericType};

type DotConfig = NumericConfig;

pub struct DotChip<F: PrimeField> {
    pub config: Rc<DotConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DotChip<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, numeric_config: NumericConfig) -> DotConfig {
        let selector = meta.selector();
        let columns = &numeric_config.columns;

        meta.create_gate("dot gate", |meta| {
            let s = meta.query_selector(selector);

            let num_inputs = (columns.len() - 1) / 2;
            let gate_inputs = columns[0..num_inputs]
                .to_vec()
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::cur()))
                .collect::<Vec<_>>();
            let gate_weights = columns[num_inputs..2 * num_inputs]
                .to_vec()
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::cur()))
                .collect::<Vec<_>>();
            let gate_outputs = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

            let res = gate_inputs
                .iter()
                .zip(gate_weights)
                .map(|(a, b)| a.clone() * b.clone())
                .fold(Expression::Constant(F::ZERO), |a, b| a + b);

            vec![s * (res - gate_outputs)]
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Dot, vec![selector]);

        DotConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }

    pub fn get_input_columns(config: &DotConfig) -> Vec<Column<Advice>> {
        let num_inputs = (config.columns.len() - 1) / 2;
        config.columns[0..num_inputs].to_vec()
    }

    pub fn get_weight_columns(config: &DotConfig) -> Vec<Column<Advice>> {
        let num_inputs = (config.columns.len() - 1) / 2;
        config.columns[num_inputs..(2 * num_inputs)].to_vec()
    }
}

impl<F: PrimeField> Numeric<F> for DotChip<F> {
    fn name(&self) -> String {
        "Dot".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        self.config.columns.len()
    }

    fn num_input_cols_per_row(&self) -> usize {
        (self.config.columns.len() - 1) / 2
    }

    fn compute_row(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check input and weight shapes
        assert_eq!(inputs.len(), 2);
        let input = &inputs[0];
        let weight = &inputs[1];
        assert_eq!(input.len(), weight.len());
        assert_eq!(input.len(), self.num_input_cols_per_row());

        // Enable selectors
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Dot).unwrap()[0];
            selector.enable(region, row_offset)?;
        }

        // Assign input and weight columns by copy advice
        let inp_cols = DotChip::<F>::get_input_columns(&self.config);
        let input = input
            .iter()
            .enumerate()
            .map(|(i, cell)| cell.copy_advice(|| "", region, inp_cols[i], row_offset))
            .collect::<Result<Vec<_>, _>>()?;
        let weight_cols = DotChip::<F>::get_weight_columns(&self.config);
        let weight = weight
            .iter()
            .enumerate()
            .map(|(i, cell)| cell.copy_advice(|| "", region, weight_cols[i], row_offset))
            .collect::<Result<Vec<_>, _>>()?;

        // All columns need to be assigned include the blank column
        // This use zero to fill the blank column
        let zero = constants[0];
        if self.config.columns.len() % 2 == 0 {
            zero.copy_advice(
                || "",
                region,
                self.config.columns[self.config.columns.len() - 2],
                row_offset,
            )?;
        }

        // Calculate the dot product
        let value = input
            .iter()
            .zip(weight.iter())
            .map(|(a, b)| a.value().copied() * b.value())
            .reduce(|a, b| a + b)
            .unwrap();

        // Assign the output column
        let res = region.assign_advice(
            || "",
            self.config.columns[self.config.columns.len() - 1],
            row_offset,
            || value,
        )?;

        Ok(vec![res])
    }

    fn compute(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check input and weight shapes
        let mut input = inputs[0].clone();
        let mut weight = inputs[1].clone();
        assert_eq!(input.len(), weight.len());
        let cols_per_row = self.num_input_cols_per_row();
        let zero = constants[0];

        // Fill the input and weight columns to be multiple of num_input_cols_per_row
        while input.len() % cols_per_row != 0 {
            input.push(&zero);
            weight.push(&zero);
        }

        // Assign input and weight columns
        let outputs = self.compute_rows(
            layouter.namespace(|| "dot rows"),
            &vec![input, weight],
            constants,
        )?;

        // Use accumulator to sum up all outputs
        let acc_chip = AccumulatorChip::<F>::construct(self.config.clone());
        Ok(acc_chip.compute(
            layouter.namespace(|| "dot adder"),
            &vec![outputs.iter().collect::<Vec<_>>()],
            &vec![zero],
        )?)
    }
}
