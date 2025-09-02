use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, ShapeError};

use crate::{
    numeric::{
        div_same::DivSameLayouter, mul_same::MulSameLayouter, nonlinear::exp::ExpLookUp,
        sum::SumLayouter, NumericConfig, NumericConsumer, NumericLayout, NumericType,
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{exp, Int},
    },
};

use super::Operation;

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
        _attributes: &BTreeMap<String, Vec<f64>>,
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
        _attributes: &BTreeMap<String, Vec<f64>>,
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
        constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let input = inputs[0].clone();

        // Get constants
        // let constants = self.get_default_constants(constants);
        let constants = self.get_constants(
            constants,
            vec![0, 1, self.numeric_config.scale_factor as Int],
        );

        let exp = ExpLookUp::<F>::construct(self.numeric_config.clone());
        let div_same = DivSameLayouter::<F>::construct(self.numeric_config.clone());
        let sum = SumLayouter::<F>::construct(self.numeric_config.clone());
        let mul_same = MulSameLayouter::<F>::construct(self.numeric_config.clone());

        let mut row_offset = 0;

        let output = layouter
            .assign_region(
                || "softmax",
                |mut region| {
                    let region = &mut region;
                    let output = input
                        .outer_iter()
                        .map(|input| {
                            let customise_row_offset = row_offset;
                            // TODO: need to be optimized
                            // In this chip we can use exp and div layouter to reduce some constraints of copy advice
                            // It can reduce rows from x to y per unit
                            let exp_out = match exp.layout(
                                region,
                                customise_row_offset,
                                &vec![input.iter().map(|x| x.as_ref()).collect::<Vec<_>>()],
                                &constants,
                            ) {
                                Ok(output) => output,
                                Err(_) => panic!("Exp compute failed"),
                            };
                            row_offset = exp_out.1;
                            let exp_out = exp_out.0;

                            let exp_sum = match sum.layout(
                                region,
                                row_offset,
                                &vec![exp_out.iter().collect()],
                                &constants,
                            ) {
                                Ok(output) => output,
                                Err(_) => panic!("Sum compute failed"),
                            };
                            row_offset = exp_sum.1;
                            let exp_sum = &exp_sum.0[0];

                            let mul_same_out = match mul_same.layout(
                                region,
                                row_offset,
                                &vec![exp_out.iter().collect(), vec![constants[2]]],
                                &constants,
                            ) {
                                Ok(output) => output,
                                Err(_) => panic!("MulSame compute failed"),
                            };
                            row_offset = mul_same_out.1;
                            let mul_same_out = mul_same_out.0;

                            div_same
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![mul_same_out.iter().collect(), vec![&exp_sum]],
                                    &constants,
                                )
                                .unwrap()
                                .0
                        })
                        .flatten()
                        .collect::<Vec<_>>();

                    Ok(output)
                },
            )
            .unwrap();

        Ok(vec![Array::from_shape_vec(
            input.shape(),
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
        let input = inputs[0].clone();
        let gradient = input.iter().map(|x| x.clone()).collect::<Vec<_>>();
        Ok(vec![Array::from_shape_vec(input.shape(), gradient)?])
    }
}

impl<F: PrimeField> NumericConsumer for SoftMaxChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::Exp,
            NumericType::Sum,
            NumericType::DivSame,
            NumericType::MulSame,
        ]
    }
}
