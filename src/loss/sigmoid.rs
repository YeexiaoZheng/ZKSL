use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, ShapeError};

use super::Loss;
use crate::numeric::div::DivLayouter;
use crate::numeric::mul_same::MulSameLayouter;
use crate::numeric::NumericLayout;
use crate::{
    numeric::{
        add_same::AddSameLayouter, div_same::DivSameLayouter, nonlinear::exp::ExpLookUp,
        square::SquareLayouter, sub::SubLayouter, NumericConfig, NumericConsumer, NumericType,
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{exp, ln, Int},
    },
};

pub struct SigmoidCrossEntropyLossChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> SigmoidCrossEntropyLossChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    // 计算期望的输出和梯度
    pub fn compute(
        input: &Tensor,
        label: &Vec<Int>,
        numeric_config: &NumericConfig,
    ) -> Result<(Int, Tensor), ShapeError> {
        // 检查输入形状
        assert_eq!(input.ndim(), 2);
        assert_eq!(input.shape()[0], label.len());
        assert_eq!(label.len(), numeric_config.batch_size);

        let n = input.shape()[0] as Int;

        // 计算sigmoid: 1/(1 + e^(-x))
        let neg_x = input.mapv(|x| -x);
        let e_neg_x = neg_x.mapv(|x| exp(x, numeric_config.scale_factor));
        let denominator = e_neg_x + (numeric_config.scale_factor as Int);
        let sigmoid = denominator.mapv(|x| {
            (numeric_config.scale_factor as Int * numeric_config.scale_factor as Int) / x
        });

        // 计算 log(sigmoid) 和 log(1-sigmoid)
        let log_sigmoid = sigmoid.mapv(|x| ln(x, numeric_config.scale_factor));
        let log_one_minus_sigmoid = sigmoid.mapv(|x| {
            ln(
                numeric_config.scale_factor as Int - x,
                numeric_config.scale_factor,
            )
        });

        // 计算损失
        let mut loss = 0;
        for (i, &y) in label.iter().enumerate() {
            if y == 1 {
                loss -= log_sigmoid[[i, 0]];
            } else {
                loss -= log_one_minus_sigmoid[[i, 0]];
            }
        }

        // 计算梯度 dp = p - y
        let mut dsigmoid = sigmoid.clone();
        for (i, &y) in label.iter().enumerate() {
            dsigmoid[[i, 0]] -= y * (numeric_config.scale_factor as Int);
        }

        Ok((loss / n, dsigmoid / n))
    }
}

impl<F: PrimeField> Loss<F> for SigmoidCrossEntropyLossChip<F> {
    fn compute(
        &self,
        mut layouter: impl Layouter<F>,
        input: &AssignedTensorRef<F>,
        label: &AssignedTensorRef<F>,
        constants: &BTreeMap<Int, CellRc<F>>,
    ) -> Result<AssignedTensor<F>, ShapeError> {
        let exp = ExpLookUp::<F>::construct(self.numeric_config.clone());
        let add_same = AddSameLayouter::<F>::construct(self.numeric_config.clone());
        let sub = SubLayouter::<F>::construct(self.numeric_config.clone());
        let div = DivLayouter::<F>::construct(self.numeric_config.clone());
        let div_same = DivSameLayouter::<F>::construct(self.numeric_config.clone());
        let mul_same = MulSameLayouter::<F>::construct(self.numeric_config.clone());
        let square = SquareLayouter::<F>::construct(self.numeric_config.clone());

        let sf = constants
            .get(&(self.numeric_config.scale_factor as Int))
            .unwrap()
            .clone();

        let bs = constants
            .get(&(self.numeric_config.batch_size as Int))
            .unwrap()
            .clone();

        // 使用 MulSameLayouter 计算缩放因子的平方
        let constants = self.get_default_constants(constants); // 移到这里
        let sf_squared = layouter
            .assign_region(
                || "compute scale factor squared",
                |mut region| {
                    let square_out = square
                        .layout(&mut region, 0, &vec![vec![sf.as_ref()]], &constants)
                        .unwrap();
                    Ok(square_out.0[0].clone())
                },
            )
            .unwrap();

        let output = layouter
            .assign_region(
                || "sigmoid cross entropy loss",
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;

                    // 计算-x，然后计算e^(-x)
                    let neg_input = input
                        .iter()
                        .map(|x| {
                            let dsigmoid = sub
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![vec![constants[0]], vec![x.as_ref()]],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = dsigmoid.1;
                            dsigmoid.0[0].clone()
                        })
                        .collect::<Vec<_>>();

                    let exp_out = exp
                        .layout(
                            region,
                            row_offset,
                            &vec![neg_input.iter().collect()],
                            &constants,
                        )
                        .unwrap();
                    row_offset = exp_out.1;
                    let exp_out = exp_out.0;

                    let add_out = add_same
                        .layout(
                            region,
                            row_offset,
                            &vec![exp_out.iter().collect(), vec![sf.as_ref()]],
                            &constants,
                        )
                        .unwrap();
                    row_offset = add_out.1;
                    let add_out = add_out.0;

                    // 创建与输入等长的sf_squared向量
                    let sf_squared_vec = add_out.iter().map(|_| &sf_squared).collect::<Vec<_>>();
                    let sigmoid = div
                        .layout(
                            region,
                            row_offset,
                            &vec![sf_squared_vec, add_out.iter().collect()],
                            &constants,
                        )
                        .unwrap();
                    row_offset = sigmoid.1;
                    let sigmoid = sigmoid.0;

                    // 计算梯度 p - y
                    let output = label
                        .iter()
                        .zip(sigmoid.iter())
                        .map(|(y, p)| {
                            let y_times_sf = mul_same
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![vec![y.as_ref()], vec![sf.as_ref()]],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = y_times_sf.1;
                            let y_times_sf = y_times_sf.0[0].clone();

                            let dsigmoid = sub
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![vec![p], vec![&y_times_sf]],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = dsigmoid.1;
                            dsigmoid.0[0].clone()
                        })
                        .collect::<Vec<_>>();

                    // 除以batch size
                    let output = div_same
                        .layout(
                            region,
                            row_offset,
                            &vec![output.iter().collect(), vec![bs.as_ref()]],
                            &constants,
                        )
                        .unwrap();

                    Ok(output.0)
                },
            )
            .unwrap();

        Ok(Array::from_shape_vec(
            input.shape(),
            output.into_iter().map(|x| Rc::new(x)).collect(),
        )?)
    }
}

impl<F: PrimeField> NumericConsumer for SigmoidCrossEntropyLossChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::Exp,
            NumericType::AddSame,
            NumericType::Sub,
            NumericType::Div,
            NumericType::DivSame,
            NumericType::MulSame,
            NumericType::Square,
        ]
    }
}
