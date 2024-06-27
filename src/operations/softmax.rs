use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, ShapeError};

use crate::{
    numerics::{
        accumulator,
        nonlinear::{exp::ExpChip, nonlinear::NonLinearNumeric},
        numeric::{Numeric, NumericConfig, NumericType},
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor, NUMERIC_CONFIG},
        math::exp,
    },
};

use super::operation::{NumericConsumer, Operation};

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
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let input = &inputs[0];
        let scale_factor = NUMERIC_CONFIG.lock().unwrap().scale_factor;

        let exp_out = input
            .clone()
            .into_iter()
            .map(|x| exp(x, scale_factor))
            .collect::<Vec<_>>();

        let exp_sum = exp_out.iter().sum::<i64>();

        Ok(vec![Array::from_shape_vec(
            input.shape(),
            exp_out
                .clone()
                .into_iter()
                .map(|x| ((x as f64) / (exp_sum as f64) * (scale_factor as f64)) as i64)
                .collect::<Vec<_>>(),
        )?])
    }
}

impl<F: PrimeField> Operation<F> for SoftMaxChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &HashMap<i64, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let input = inputs[0].clone();

        let exp_chip = ExpChip::<F>::construct(self.numeric_config.clone());
        let acc_chip = accumulator::AccumulatorChip::<F>::construct(self.numeric_config.clone());

        let exp_out = match NonLinearNumeric::forward(
            &exp_chip,
            layouter.namespace(|| "Exp forward"),
            &vec![input.iter().map(|x| x.as_ref()).collect::<Vec<_>>()],
            &vec![],
        ) {
            Ok(output) => output,
            Err(_) => panic!("Exp forward failed"),
        };

        let exp_sum = match Numeric::forward(
            &acc_chip,
            layouter.namespace(|| "Accumulator forward"),
            &vec![exp_out.iter().map(|x| x).collect::<Vec<_>>()],
            &vec![],
        ) {
            Ok(output) => output,
            Err(_) => panic!("Accumulator forward failed"),
        };
        let _exp_sum = &exp_sum[0];

        Ok(vec![])
    }

    fn backward(
        &self,
        _layouter: impl Layouter<F>,
        _inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &HashMap<i64, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        Ok(vec![])
    }
}

impl<F: PrimeField> NumericConsumer for SoftMaxChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![NumericType::FieldLookUp, NumericType::Exp]
    }
}
