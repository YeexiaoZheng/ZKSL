use std::{collections::BTreeMap, marker::PhantomData, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{s, Array, Axis, IxDyn, ShapeError};

use crate::{
    numeric::{
        div_same::DivSameLayouter, sum::SumLayouter, NumericConfig, NumericConsumer, NumericLayout,
        NumericType,
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::Int,
    },
};

use super::Operation;

#[derive(Clone, Debug, Default)]
pub struct MeanChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> MeanChip<F> {
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
        let output = input.mean_axis(Axis(1)).unwrap();
        Ok(vec![output.into_dyn()])
    }

    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        // let input = &inputs[0];
        //
        // Ok(vec![input.clone()])
        let gradient = &inputs[0]; // 上层传来的梯度
        let input = &inputs[1]; // 原始输入，用于获取形状

        // 创建与输入相同形状的梯度张量
        let mut output = Array::zeros(input.shape());

        let n = input.shape()[1] as Int; // 使用 Int 类型

        // 对每个位置的梯度进行扩展
        for i in 0..input.shape()[0] {
            for k in 0..input.shape()[2] {
                for j in 0..input.shape()[1] {
                    // 确保类型一致性，使用 Int 类型
                    output[[i, j, k]] = gradient[[i, k]] / n;
                }
            }
        }

        Ok(vec![output.into_dyn()])
    }
}

impl<F: PrimeField> Operation<F> for MeanChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let input = inputs[0].clone();
        let zero = constants.get(&0).unwrap().clone();
        let one = constants.get(&1).unwrap().clone();

        let sum = SumLayouter::<F>::construct(self.numeric_config.clone());
        let div_same = DivSameLayouter::<F>::construct(self.numeric_config.clone());

        let output = layouter
            .assign_region(
                || "mean",
                |mut region| {
                    let mut output = vec![];
                    let mut row_offset = 0;
                    for i in 0..input.shape()[0] {
                        for j in 0..input.shape()[2] {
                            let sum = sum
                                .layout(
                                    &mut region,
                                    row_offset,
                                    &vec![input
                                        .slice(s![i, .., j])
                                        .iter()
                                        .map(|x| x.as_ref())
                                        .collect()],
                                    &vec![zero.as_ref()],
                                )
                                .unwrap();
                            row_offset = sum.1;
                            let sum = sum.0;
                            output.push(sum[0].clone());
                        }
                    }
                    let mean = div_same
                        .layout(
                            &mut region,
                            row_offset,
                            &vec![
                                output.iter().collect(),
                                vec![constants
                                    .get(&(input.shape()[1] as Int))
                                    .unwrap()
                                    .clone()
                                    .as_ref()],
                            ],
                            &vec![zero.as_ref(), one.as_ref()],
                        )
                        .unwrap();
                    Ok(mean.0)
                },
            )
            .unwrap();

        Ok(vec![Array::from_shape_vec(
            IxDyn(&[input.shape()[0], input.shape()[2]]),
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
        // let inpgrad = inputs[0].clone();
        // let _input = inputs[1].clone();
        //
        // Ok(vec![inpgrad.to_owned()])

        let gradient = inputs[0].clone(); // 上层传来的梯度
        let input = inputs[1].clone(); // 原始输入

        let div_same = DivSameLayouter::<F>::construct(self.numeric_config.clone());
        let zero = constants.get(&0).unwrap().clone();
        let one = constants.get(&1).unwrap().clone();
        let n = constants.get(&(input.shape()[1] as Int)).unwrap().clone();

        // 在电路中实现梯度的扩展和缩放
        let output = layouter
            .assign_region(
                || "mean_backward",
                |mut region| {
                    let mut output = vec![];
                    let mut row_offset = 0;

                    // 对每个位置的梯度进行处理
                    for i in 0..input.shape()[0] {
                        for _ in 0..input.shape()[1] {
                            for k in 0..input.shape()[2] {
                                // 缩放梯度值
                                let scaled = div_same
                                    .layout(
                                        &mut region,
                                        row_offset,
                                        &vec![
                                            vec![gradient
                                                .slice(s![i, k])
                                                .iter()
                                                .next()
                                                .unwrap()
                                                .as_ref()],
                                            vec![n.as_ref()],
                                        ],
                                        &vec![zero.as_ref(), one.as_ref()],
                                    )
                                    .unwrap();
                                row_offset = scaled.1;
                                output.push(scaled.0[0].clone());
                            }
                        }
                    }
                    Ok((output, row_offset))
                },
            )
            .unwrap();

        // 重塑为原始输入的形状
        Ok(vec![Array::from_shape_vec(
            IxDyn(&[input.shape()[0], input.shape()[1], input.shape()[2]]),
            output.0.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>(),
        )?])
    }
}

impl<F: PrimeField> NumericConsumer for MeanChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![NumericType::Sum, NumericType::DivSame]
    }
}
