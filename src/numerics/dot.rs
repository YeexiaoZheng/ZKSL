use std::marker::PhantomData;

use halo2_proofs::{
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Expression},
    poly::Rotation,
};

use crate::layers::layer::AssignedTensor;

use super::numeric::{Numeric, NumericConfig, NumericType, _NumericConfig};

type DotConfig = _NumericConfig;

pub struct DotChip<F: PrimeField> {
    config: DotConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DotChip<F> {
    pub fn construct(config: DotConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: _NumericConfig,
    ) -> _NumericConfig {
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
            let gate_weights = columns[num_inputs..columns.len() - 1]
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

        _NumericConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> Numeric<F> for DotChip<F> {
    fn name(&self) -> String {
        "Dot".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        self.config.columns.len()
    }

    fn forward(&self, inputs: Vec<AssignedTensor<F>>) -> Result<Vec<AssignedTensor<F>>, halo2_proofs::plonk::Error> {
        let mut outputs = vec![];
        for input in inputs.iter() {
            let output = input
                .iter()
                .map(|cell| cell.value())
                .collect::<Vec<_>>();
            outputs.push(output);
        }
        Ok(inputs)
    }

    fn expose_forward(
        &self,
        layouter: impl halo2_proofs::circuit::Layouter<F>,
        forward_outputs: &Vec<AssignedTensor<F>>,
        row: usize,
    ) -> Result<(), halo2_proofs::plonk::ErrorFront> {
        // layouter.constrain_instance(forward_output.0, self.config.public, row)
        todo!()
    }
}
