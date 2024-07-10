use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, ShapeError};

use crate::{
    numerics::{
        nonlinear::relu::ReluChip,
        numeric::{Numeric, NumericConfig, NumericConsumer, NumericType},
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{relu, Int},
    },
};

use super::operation::Operation;

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
        _numeric_config: &NumericConfig,
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

    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
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
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &HashMap<Int, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let input = inputs[0].clone();
        let zero = constants.get(&0).unwrap().clone();
        let one = constants.get(&1).unwrap().clone();

        let relu_chip = ReluChip::<F>::construct(self.numeric_config.clone());

        let output = match relu_chip.forward(
            layouter.namespace(|| "ReLU forward"),
            &vec![input.iter().map(|x| x.as_ref()).collect::<Vec<_>>()],
            &vec![zero.as_ref(), one.as_ref()],
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
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &HashMap<Int, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let inpgrad = inputs[0].clone();
        let zero = constants.get(&0).unwrap().clone();
        let one = constants.get(&1).unwrap().clone();

        let relu_chip = ReluChip::<F>::construct(self.numeric_config.clone());

        let outgrad = match relu_chip.forward(
            layouter.namespace(|| "ReLU backward"),
            &vec![inpgrad.iter().map(|x| x.as_ref()).collect()],
            &vec![zero.as_ref(), one.as_ref()],
        ) {
            Ok(output) => output,
            Err(_) => panic!("ReLU backward failed"),
        };
        Ok(vec![Array::from_shape_vec(
            inpgrad.shape(),
            outgrad.into_iter().map(|x| Rc::new(x)).collect(),
        )?])
    }
}

impl<F: PrimeField> NumericConsumer for ReLUChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![NumericType::FieldLookUp, NumericType::Relu]
    }
}
