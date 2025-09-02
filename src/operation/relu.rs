use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, ShapeError};

use crate::{
    numeric::{
        nonlinear::relu::ReluLookUp, NumericConfig, NumericConsumer, NumericLayout, NumericType,
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{relu, Int},
    },
};

use super::Operation;

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
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let input = &inputs[0];
        // println!("relu input shape: {:?}", input.shape());

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
        _attributes: &BTreeMap<String, Vec<f64>>,
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
        constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0].clone();
        let constants = self.get_default_constants(constants);

        let relu = ReluLookUp::<F>::construct(self.numeric_config.clone());

        let output = layouter
            .assign_region(
                || "ReLU forward",
                |mut region| {
                    let row_offset = 0;
                    let input = input.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                    match relu.layout(&mut region, row_offset, &vec![input], &constants) {
                        Ok(output) => Ok(output.0),
                        Err(_) => panic!("ReLU forward failed"),
                    }
                },
            )
            .unwrap();

        Ok(vec![Array::from_shape_vec(
            input.shape(),
            output.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>(),
        )?])
    }

    fn backward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let inpgrad = inputs[0].clone();
        let constants = self.get_default_constants(constants);

        let relu = ReluLookUp::<F>::construct(self.numeric_config.clone());

        let outgrad = layouter
            .assign_region(
                || "ReLU backward",
                |mut region| {
                    let row_offset = 0;
                    let inpgrad = inpgrad.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                    match relu.layout(&mut region, row_offset, &vec![inpgrad], &constants) {
                        Ok(output) => Ok(output.0),
                        Err(_) => panic!("ReLU backward failed"),
                    }
                },
            )
            .unwrap();

        Ok(vec![Array::from_shape_vec(
            inpgrad.shape(),
            outgrad.into_iter().map(|x| Rc::new(x)).collect(),
        )?])
    }
}

impl<F: PrimeField> NumericConsumer for ReLUChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![NumericType::Relu]
    }
}
