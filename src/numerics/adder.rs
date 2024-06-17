use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error, Expression},
    poly::Rotation,
};

use super::numeric::{Numeric, NumericType, _NumericConfig};

type AdderConfig = _NumericConfig;

pub struct AdderChip<F: PrimeField> {
    pub config: Rc<AdderConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> AdderChip<F> {
    pub fn construct(config: Rc<AdderConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: _NumericConfig,
    ) -> AdderConfig {
        let selector = meta.selector();
        let columns = numeric_config.columns;

        meta.create_gate("adder gate", |meta| {
            let s = meta.query_selector(selector);

            let gate_inp = columns[0..columns.len() - 1]
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::cur()))
                .collect::<Vec<_>>();
            let gate_output = meta.query_advice(*columns.last().unwrap(), Rotation::cur());

            let res = gate_inp
                .iter()
                .fold(Expression::Constant(F::ZERO), |a, b| a + b.clone());

            vec![s * (res - gate_output)]
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Adder, vec![selector]);

        AdderConfig {
            columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> Numeric<F> for AdderChip<F> {
    fn name(&self) -> String {
        "Adder".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        self.config.columns.len()
    }

    fn num_input_cols_per_row(&self) -> usize {
        self.config.columns.len() - 1
    }

    fn op_row_region(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        _single_inputs: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        assert_eq!(vec_inputs.len(), 1);
        let inp = &vec_inputs[0];

        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Adder).unwrap()[0];
            selector.enable(region, row_offset)?;
        }

        inp.iter()
            .enumerate()
            .map(|(i, cell)| cell.copy_advice(|| "", region, self.config.columns[i], row_offset))
            .collect::<Result<Vec<_>, _>>()?;

        let e = inp.iter().fold(Value::known(F::ZERO), |a, b| {
            a + b.value().map(|x: &F| x.to_owned())
        });
        let res = region.assign_advice(
            || "",
            *self.config.columns.last().unwrap(),
            row_offset,
            || e,
        )?;

        Ok(vec![res])
    }

    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        assert_eq!(constants.len(), 1);

        let mut input = inputs[0].clone();
        let zero = constants[0];

        while input.len() % self.num_input_cols_per_row() != 0 {
            input.push(&zero);
        }

        let mut outputs = self.op_aligned_rows(
            layouter.namespace(|| "adder forward"),
            &vec![input],
            constants,
        )?;

        while outputs.len() != 1 {
            while outputs.len() % self.num_input_cols_per_row() != 0 {
                outputs.push(zero.clone());
            }
            let tmp = outputs.iter().map(|x| x).collect::<Vec<_>>();
            outputs = self.op_aligned_rows(
                layouter.namespace(|| "adder forward"),
                &vec![tmp],
                constants,
            )?;
        }

        Ok(outputs)
    }
}
