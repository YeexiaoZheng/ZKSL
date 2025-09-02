use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{concatenate, Axis, ShapeError};

use crate::{
    numeric::{NumericConfig, NumericConsumer, NumericType},
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::Int,
    },
};

use super::Operation;

#[derive(Clone, Debug, Default)]
pub struct ConcatChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ConcatChip<F> {
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
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let axis = match attributes.get("axis") {
            Some(x) => x[0] as usize,
            None => panic!("attributes not found!"),
        };
        Ok(vec![concatenate(
            Axis(axis),
            inputs
                .iter()
                .map(|x| x.view())
                .collect::<Vec<_>>()
                .as_slice(),
        )?])
    }

    // This function is used for non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let gradient = &inputs[0]; // 上层传来的梯度
        let mut result = Vec::new();
        let mut start_idx = 0;

        // 获取拼接的轴
        let axis = match attributes.get("axis") {
            Some(x) => x[0] as usize,
            None => panic!("attributes not found!"),
        };

        // 从第二个输入开始是原始输入的形状信息
        for input in inputs.iter().skip(1) {
            // 计算在拼接轴上的切片范围
            let end_idx = start_idx + input.shape()[axis];

            // 提取对应的梯度部分
            let grad_part = gradient.slice_axis(
                Axis(axis),
                ndarray::Slice::new(start_idx as isize, Some(end_idx as isize), 1),
            );
            result.push(grad_part.to_owned());

            start_idx = end_idx;
        }

        Ok(result)
    }
}

impl<F: PrimeField> Operation<F> for ConcatChip<F> {
    fn forward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // TODO: hacker: concatenate is not in circuit!
        let axis = match attributes.get("axis") {
            Some(x) => x[0] as usize,
            None => panic!("attributes not found!"),
        };
        Ok(vec![concatenate(
            Axis(axis),
            inputs
                .iter()
                .map(|x| x.view())
                .collect::<Vec<_>>()
                .as_slice(),
        )?])
    }

    fn backward(
        &self,
        _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let gradient = inputs[0].clone(); // 上层梯度
        let mut result = Vec::new();
        let mut start_idx = 0;

        // 获取拼接的轴
        let axis = match attributes.get("axis") {
            Some(x) => x[0] as usize,
            None => panic!("attributes not found!"),
        };

        // 从第二个输入开始是原始输入的形状信息
        for input in inputs.iter().skip(1) {
            // 计算在拼接轴上的切片范围
            let end_idx = start_idx + input.shape()[axis];

            // 提取对应的梯度部分
            let grad_part = gradient.slice_axis(
                Axis(axis),
                ndarray::Slice::new(start_idx as isize, Some(end_idx as isize), 1),
            );
            result.push(grad_part.to_owned());

            start_idx = end_idx;
        }

        Ok(result)
    }
}

impl<F: PrimeField> NumericConsumer for ConcatChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![]
    }
}
