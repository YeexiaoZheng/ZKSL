use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{s, Array, IxDyn, ShapeError};

use crate::{
    numerics::{
        dot::DotChip,
        numeric::{Numeric, NumericConfig, NumericType},
    },
    utils::helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
};

use super::operation::{NumericConsumer, Operation};

#[derive(Clone, Debug, Default)]
pub struct GemmChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> GemmChip<F> {
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
        let weight = &inputs[1];
        let input_shape = (input.shape()[0], input.shape()[1]);
        let weight_shape = (weight.shape()[0], weight.shape()[1]);
        assert_eq!(input_shape.1, weight_shape.0);

        let input = input.clone().into_shape(input_shape)?;
        let weight = weight.clone().into_shape(weight_shape)?;

        Ok(vec![input.dot(&weight).into_dyn()])
    }

    // This function is used for non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let input = &inputs[0];
        let weight = &inputs[1];
        let input_shape = (input.shape()[0], input.shape()[1]);
        let weight_shape = (weight.shape()[0], weight.shape()[1]);
        assert_eq!(input_shape.1, weight_shape.0);

        let input = input.clone().into_shape(input_shape)?;
        let weight = weight.clone().into_shape(weight_shape)?;

        Ok(vec![input.dot(&weight).into_dyn()])
    }
}

impl<F: PrimeField> Operation<F> for GemmChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &HashMap<i64, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // Check input shape
        let input = inputs[0].clone();
        let weight = inputs[1].clone();
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        assert_eq!(input_shape.len(), 2);
        assert_eq!(input_shape.len(), weight_shape.len());
        assert_eq!(input_shape[1], weight_shape[0]);

        // Get constants
        let zero = constants.get(&0).unwrap().clone();
        let constants = vec![zero.as_ref()];

        // Initialize dot chip
        let dot_chip = DotChip::construct(self.numeric_config.clone());

        // Forward pass
        let mut outputs = vec![];
        for i in 0..input_shape[0] {
            for j in 0..weight_shape[1] {
                let input = input
                    .slice(s![i, ..])
                    .into_iter()
                    .map(|x| x.as_ref())
                    .collect::<Vec<_>>();
                let weight = weight
                    .slice(s![.., j])
                    .into_iter()
                    .map(|x| x.as_ref())
                    .collect::<Vec<_>>();
                outputs.extend(
                    match dot_chip.forward(
                        layouter.namespace(|| format!("dot_{}_{}", i, j)),
                        &vec![input, weight],
                        &constants,
                    ) {
                        Ok(output) => output,
                        Err(e) => panic!("Error in GemmChip.dot_chip: {:?}", e),
                    },
                );
            }
        }

        Ok(vec![Array::from_shape_vec(
            IxDyn(&[input_shape[0], weight_shape[1]]),
            outputs.into_iter().map(|x| Rc::new(x)).collect(),
        )?])
    }
    
    fn backward(
        &self,
        _layouter: impl Layouter<F>,
        _inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &HashMap<i64, CellRc<F>>,
        _attributes: &HashMap<String, f64>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        todo!()
    }
}

impl<F: PrimeField> NumericConsumer for GemmChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![NumericType::Dot, NumericType::Accumulator]
    }
}
