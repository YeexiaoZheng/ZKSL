use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, IxDyn, ShapeError};

use crate::{
    numeric::{add::AddLayouter, NumericConfig, NumericConsumer, NumericLayout, NumericType},
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::Int,
    },
};

use super::Operation;

#[derive(Clone, Debug, Default)]
pub struct AddChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> AddChip<F> {
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
        assert_eq!(inputs.len(), 2);
        let input1 = &inputs[0];
        let input2 = &inputs[1];
        let output = input1 + input2;

        Ok(vec![output])
    }

    // This function is used for non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let inpgrad = &inputs[0].clone();

        Ok(vec![inpgrad.clone(), inpgrad.clone()])
    }
}

impl<F: PrimeField> Operation<F> for AddChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // Check input shape
        assert_eq!(inputs.len(), 2);
        let input1 = inputs[0].clone();
        let input2 = inputs[1].clone();
        assert_eq!(input1.shape(), input2.shape());

        // Get constants
        let constants = self.get_default_constants(constants);

        // Initialize numeric chip
        let add = AddLayouter::construct(self.numeric_config.clone());

        // Forward pass
        let output = layouter
            .assign_region(
                || "add operation",
                |mut region| {
                    Ok(add
                        .layout(
                            &mut region,
                            0,
                            &vec![
                                input1.iter().map(|x| x.as_ref()).collect(),
                                input2.iter().map(|x| x.as_ref()).collect(),
                            ],
                            &constants,
                        )
                        .unwrap()
                        .0)
                },
            )
            .unwrap();

        Ok(vec![Array::from_shape_vec(
            IxDyn(input1.shape()),
            output.into_iter().map(|x| Rc::new(x)).collect(),
        )?])
    }

    fn backward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // Check input shape
        // assert_eq!(inputs.len(), 3);
        let inpgrad = inputs[0].clone();

        Ok(vec![inpgrad.to_owned(), inpgrad.to_owned()])
    }
}

impl<F: PrimeField> NumericConsumer for AddChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![NumericType::Add]
    }
}
