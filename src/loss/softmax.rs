use std::{collections::BTreeMap, marker::PhantomData, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, Axis, ShapeError};

use super::Loss;
use crate::{
    numeric::{
        div_same::DivSameLayouter, max::MaxLayouter, mul_same::MulSameLayouter,
        nonlinear::exp::ExpLookUp, sub::SubLayouter, sub_same::SubSameLayouter, sum::SumLayouter,
        NumericConfig, NumericConsumer, NumericLayout, NumericType,
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{exp, ln, Int},
    },
};

pub struct SoftMaxLossChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> SoftMaxLossChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    pub fn compute(
        input: &Tensor,
        label: &Vec<Int>,
        numeric_config: &NumericConfig,
    ) -> Result<(Int, Tensor), ShapeError> {
        // Check input shape
        assert_eq!(input.ndim(), 2);
        assert_eq!(input.shape()[0], label.len());
        assert_eq!(label.len(), numeric_config.batch_size);

        let n = input.shape()[0] as Int;
        let max = Array::from_shape_vec(
            input.shape(),
            input
                .outer_iter()
                .map(|row| row.iter().fold(0, |acc, &x| acc.max(x)))
                .map(|max| vec![max; input.shape()[1]])
                .flatten()
                .collect::<Vec<_>>(),
        )?;
        let f = input - max;
        let ef = f.mapv(|x| exp(x, numeric_config.scale_factor));
        let efs = ef.sum_axis(Axis(1));

        let l: Int = f
            .outer_iter()
            .zip(label.iter())
            .map(|(row, &y)| row[y as usize])
            .sum();
        let loss = -l + efs.mapv(|x| ln(x, numeric_config.scale_factor)).sum();

        let efs = Array::from_shape_vec(
            input.shape(),
            efs.iter()
                .map(|&x| vec![x; input.shape()[1]])
                .flatten()
                .collect::<Vec<_>>(),
        )?;
        let mut dscore = ef * (numeric_config.scale_factor as Int) / efs;
        for (i, y) in label.iter().enumerate() {
            dscore[[i, *y as usize]] -= numeric_config.scale_factor as Int;
        }

        Ok((loss / n, dscore / n))
    }
}

impl<F: PrimeField> Loss<F> for SoftMaxLossChip<F> {
    fn compute(
        &self,
        mut layouter: impl Layouter<F>,
        input: &AssignedTensorRef<F>,
        label: &AssignedTensorRef<F>,
        constants: &BTreeMap<Int, CellRc<F>>,
    ) -> Result<AssignedTensor<F>, ShapeError> {
        let exp = ExpLookUp::<F>::construct(self.numeric_config.clone());
        let max = MaxLayouter::<F>::construct(self.numeric_config.clone());
        let sub = SubLayouter::<F>::construct(self.numeric_config.clone());
        let sub_same = SubSameLayouter::<F>::construct(self.numeric_config.clone());
        let mul_same = MulSameLayouter::<F>::construct(self.numeric_config.clone());
        let div_same = DivSameLayouter::<F>::construct(self.numeric_config.clone());
        let sum = SumLayouter::<F>::construct(self.numeric_config.clone());

        let constants = self.get_constants(
            constants,
            vec![
                0,
                1,
                2,
                self.numeric_config.scale_factor as Int,
                self.numeric_config.batch_size as Int,
            ],
        );
        let sf = constants[3];
        let bs = constants[4];

        let dscore = layouter
            .assign_region(
                || "softmax loss",
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;

                    let output = input
                        .outer_iter()
                        .zip(label.outer_iter())
                        .map(|(row, y)| {
                            let row = row.iter().map(|x| x.as_ref()).collect::<Vec<_>>();

                            let max_out = max
                                .layout(region, row_offset, &vec![row.clone()], &constants)
                                .unwrap();
                            row_offset = max_out.1;
                            let max_out = max_out.0;

                            // f
                            let sub_out = sub_same
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![row.clone(), vec![&max_out[0]]],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = sub_out.1;
                            let sub_out = sub_out.0;

                            // ef
                            let exp_out = exp
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![sub_out.iter().collect()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = exp_out.1;
                            let exp_out = exp_out.0;

                            // efs
                            let sum_out = sum
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![exp_out.iter().collect()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = sum_out.1;
                            let sum_out = sum_out.0;

                            // ef/efs
                            let mul_same_out = mul_same
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![exp_out.iter().collect(), vec![sf]],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = mul_same_out.1;
                            let mul_same_out = mul_same_out.0;
                            let dscore = div_same
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![mul_same_out.iter().collect(), vec![&sum_out[0]]],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = dscore.1;
                            let dscore = dscore.0;

                            let subs = mul_same
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![y.iter().map(|x| x.as_ref()).collect(), vec![sf]],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = subs.1;
                            let subs = subs.0;

                            let dscore = sub
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![dscore.iter().collect(), subs.iter().collect()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = dscore.1;

                            dscore.0
                        })
                        .flatten()
                        .collect::<Vec<_>>();

                    let output = div_same
                        .layout(
                            region,
                            row_offset,
                            &vec![output.iter().collect(), vec![bs]],
                            &constants,
                        )
                        .unwrap();
                    let output = output.0;

                    Ok(output)
                },
            )
            .unwrap();

        Ok(Array::from_shape_vec(
            input.shape(),
            dscore.into_iter().map(|x| Rc::new(x)).collect(),
        )?)
    }
}

impl<F: PrimeField> NumericConsumer for SoftMaxLossChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::Exp,
            NumericType::Max,
            NumericType::Sub,
            NumericType::SubSame,
            NumericType::DivSame,
            NumericType::MulSame,
            NumericType::Sum,
        ]
    }
}
