use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, ShapeError};

use crate::{
    numerics::{
        accumulator::AccumulatorChip,
        div::DivChip,
        mul::MulChip,
        nonlinear::exp::ExpChip,
        numeric::{Numeric, NumericConfig, NumericConsumer, NumericType},
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{exp, Int},
    },
};

use super::operation::Operation;

// TODO: only implement axis = 1, to implement other axis by judge attribute

// IMPORTANT: It returns exp(x^i / scale_factor) * scale_factor / sum(exp(x^i / scale_factor) * scale_factor) * scale_factor
#[derive(Clone, Debug, Default)]
pub struct SoftMaxChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> SoftMaxChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    // This function is used for non-circuit forward
    pub fn forward(
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let input = &inputs[0];
        let scale_factor = numeric_config.scale_factor;

        let output = input
            .outer_iter()
            .map(|input| {
                let exp_out = input
                    .iter()
                    .map(|x| exp(*x, scale_factor))
                    .collect::<Vec<_>>();

                let exp_sum = exp_out.iter().sum::<Int>();

                exp_out
                    .into_iter()
                    .map(|x| ((x as f64) / (exp_sum as f64) * (scale_factor as f64)) as Int)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(vec![Array::from_shape_vec(input.shape(), output)?])
    }

    // This function is used for non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let input = &inputs[0];
        let gradient = input.iter().map(|x| x.clone()).collect::<Vec<_>>();
        Ok(vec![Array::from_shape_vec(input.shape(), gradient)?])
    }
}

impl<F: PrimeField> Operation<F> for SoftMaxChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &HashMap<Int, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let input = inputs[0].clone();

        let zero = constants.get(&0).unwrap().clone();
        let one = constants.get(&1).unwrap().clone();
        let sf = constants
            .get(&(self.numeric_config.scale_factor as Int))
            .unwrap()
            .clone();

        let exp_chip = ExpChip::<F>::construct(self.numeric_config.clone());
        let mul_chip = MulChip::<F>::construct(self.numeric_config.clone());
        let div_chip = DivChip::<F>::construct(self.numeric_config.clone());
        let acc_chip = AccumulatorChip::<F>::construct(self.numeric_config.clone());

        let output = input
            .outer_iter()
            .map(|input| {
                let exp_out = match exp_chip.compute(
                    layouter.namespace(|| "Exp compute"),
                    &vec![input.iter().map(|x| x.as_ref()).collect::<Vec<_>>()],
                    &vec![zero.as_ref(), one.as_ref()],
                ) {
                    Ok(output) => output,
                    Err(_) => panic!("Exp compute failed"),
                };

                let exp_sum = match acc_chip.compute(
                    layouter.namespace(|| "Accumulator compute"),
                    &vec![exp_out.iter().collect()],
                    &vec![zero.as_ref()],
                ) {
                    Ok(output) => output,
                    Err(_) => panic!("Accumulator compute failed"),
                };
                let exp_sum = &exp_sum[0];

                let exp_scaled = match mul_chip.compute(
                    layouter.namespace(|| "Mul compute"),
                    &vec![exp_out.iter().collect(), vec![sf.as_ref(); input.len()]],
                    &vec![zero.as_ref()],
                ) {
                    Ok(output) => output,
                    Err(_) => panic!("Mul compute failed"),
                };

                match div_chip.compute(
                    layouter.namespace(|| "Div compute"),
                    &vec![exp_scaled.iter().collect(), vec![&exp_sum; input.len()]],
                    &vec![zero.as_ref(), one.as_ref()],
                ) {
                    Ok(output) => output,
                    Err(_) => panic!("Div compute failed"),
                }
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(vec![Array::from_shape_vec(
            input.shape(),
            output.into_iter().map(|x| Rc::new(x)).collect(),
        )?])
    }

    fn backward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &HashMap<Int, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let input = inputs[0].clone();
        let gradient = input.iter().map(|x| x.clone()).collect::<Vec<_>>();
        Ok(vec![Array::from_shape_vec(input.shape(), gradient)?])
    }
}

impl<F: PrimeField> NumericConsumer for SoftMaxChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::RowLookUp,
            NumericType::Exp,
            NumericType::Mul,
            NumericType::Accumulator,
            NumericType::Div,
        ]
    }
}
