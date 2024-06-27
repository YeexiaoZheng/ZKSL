use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, ShapeError};

use crate::{
    numerics::{
        nonlinear::{nonlinear::NonLinearNumeric, relu::ReluChip},
        numeric::{NumericConfig, NumericType},
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::relu,
    },
};

use super::operation::{NumericConsumer, Operation};

#[derive(Clone, Debug, Default)]
pub struct ReLUChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ReLUChip<F> {
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

        Ok(vec![Array::from_shape_vec(
            input.shape(),
            input
                .clone()
                .into_iter()
                .map(|x| relu(x))
                .collect::<Vec<_>>(),
        )?])
    }
}

impl<F: PrimeField> Operation<F> for ReLUChip<F> {
    fn forward(
        &self,
        layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &HashMap<i64, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let reluchip = ReluChip::<F>::construct(self.numeric_config.clone());
        let input = inputs[0].clone();
        let output = match reluchip.forward(
            layouter,
            &vec![input.iter().map(|x| x.as_ref()).collect::<Vec<_>>()],
            &vec![],
        ) {
            Ok(output) => output,
            Err(_) => panic!("ReLU forward failed"),
        };
        Ok(vec![Array::from_shape_vec(
            input.shape(),
            output.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>(),
        )?])
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

impl<F: PrimeField> NumericConsumer for ReLUChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![NumericType::FieldLookUp, NumericType::Relu]
    }
}
