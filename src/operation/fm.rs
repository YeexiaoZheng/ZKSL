use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use super::Operation;
use crate::{
    numeric::{
        add::AddLayouter, div_same::DivSameLayouter, div_sf::DivSFLayouter, square::SquareLayouter,
        sub::SubLayouter, sum::SumLayouter, NumericConfig, NumericConsumer, NumericLayout,
        NumericType,
    },
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{fdiv, Int},
    },
};
use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{s, Array, Axis, IxDyn, ShapeError};

#[derive(Clone, Debug, Default)]
pub struct FMChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> FMChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    pub fn forward(
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        // 获取输入向量和权重矩阵
        let input = &inputs[0].clone(); // [batch_size, features]
        let embedding = &inputs[1].clone(); // [batch_size, features, embedding]

        // 检查并重塑维度
        let input_shape = (input.shape()[0], input.shape()[1]); // [batch_size, n_features]
        let emb_shape = (
            embedding.shape()[0],
            embedding.shape()[1],
            embedding.shape()[2],
        ); // [batch_size, features, embedding]

        // 一阶特征结果
        let input = input.to_shape(input_shape)?;
        let output_first = input.sum_axis(Axis(1));

        // 二阶部分计算
        // Sum-Square部分
        let embedding = embedding.to_shape(emb_shape)?;
        let summed_features_emb = embedding.sum_axis(Axis(1));
        let summed_features_emb_square =
            summed_features_emb.mapv(|x| fdiv(x * x, numeric_config.scale_factor as Int));

        // Square-Sum部分
        let squared_features_emb =
            embedding.mapv(|x| fdiv(x * x, numeric_config.scale_factor as Int));
        let squared_sum_features_emb = squared_features_emb.sum_axis(Axis(1));

        let y_middle = (summed_features_emb_square - squared_sum_features_emb).sum_axis(Axis(1));
        let y_second_order = y_middle.mapv(|x| x / 2);

        // 合并一阶和二阶结果
        let output = output_first + y_second_order;

        let output = output.into_shape_with_order((input_shape.0, 1))?;

        Ok(vec![output.into_dyn()])
    }

    pub fn backward(
        _inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        Ok(vec![])
    }
}

impl<F: PrimeField> Operation<F> for FMChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // 提取输入和常量
        let (input, embedding) = (inputs[0].clone(), inputs[1].clone());
        let input_shape = input.shape();
        let emb_shape = embedding.shape();

        // 2. 获取电路常量
        let zero = constants.get(&0).unwrap().clone();
        let one = constants.get(&1).unwrap().clone();
        let two = constants.get(&2).unwrap().clone();
        let constants = vec![zero.as_ref(), one.as_ref(), two.as_ref()];

        // 3. 构建所需的基础运算电路组件
        let sub = SubLayouter::construct(self.numeric_config.clone()); // 减法运算
        let div_same = DivSameLayouter::construct(self.numeric_config.clone()); // 除法运算
        let div_sf = DivSFLayouter::construct(self.numeric_config.clone()); // 除法运算
        let add = AddLayouter::construct(self.numeric_config.clone()); // 加法运算
        let sum = SumLayouter::construct(self.numeric_config.clone()); // 累加运算
        let square = SquareLayouter::construct(self.numeric_config.clone()); // 平方运算

        let output = layouter
            .assign_region(
                || "operation FM",
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;
                    let mut outputs = vec![];

                    // 对每个 batch 进行处理
                    for i in 0..input_shape[0] {
                        // 1. 计算一阶特征 sum(input)
                        let input_ = input
                            .slice(s![i, ..])
                            .into_iter()
                            .map(|x| x.as_ref())
                            .collect::<Vec<_>>();
                        let sum_input = sum
                            .layout(region, row_offset, &vec![input_], &constants)
                            .unwrap();
                        row_offset = sum_input.1;
                        let sum_input = sum_input.0;

                        // 2. 计算二阶特征
                        // 2.1 首先对 embedding 求和
                        let mut summed_features_emb = vec![];
                        for j in 0..emb_shape[2] {
                            let emb_ = embedding
                                .slice(s![i, .., j])
                                .into_iter()
                                .map(|x| x.as_ref())
                                .collect::<Vec<_>>();
                            let emb_sum = sum
                                .layout(region, row_offset, &vec![emb_], &constants)
                                .unwrap();
                            row_offset = emb_sum.1;
                            summed_features_emb.extend(emb_sum.0);
                        }

                        // 2.2 计算 summed_features_emb 的平方并除以 scale_factor
                        let summed_features_emb_square = square
                            .layout(
                                region,
                                row_offset,
                                &vec![summed_features_emb.iter().collect()],
                                &constants,
                            )
                            .unwrap();
                        row_offset = summed_features_emb_square.1;
                        let summed_features_emb_square = summed_features_emb_square.0;

                        let summed_features_emb_square = div_sf
                            .layout(
                                region,
                                row_offset,
                                &vec![summed_features_emb_square.iter().collect()],
                                &constants,
                            )
                            .unwrap();
                        row_offset = summed_features_emb_square.1;
                        let summed_features_emb_square = summed_features_emb_square.0;

                        // 2.3 计算 embedding 的平方和
                        let mut squared_sum_features_emb = vec![];
                        for j in 0..emb_shape[2] {
                            let emb_ = embedding
                                .slice(s![i, .., j])
                                .into_iter()
                                .map(|x| x.as_ref())
                                .collect::<Vec<_>>();

                            // 先计算平方
                            let squared = square
                                .layout(region, row_offset, &vec![emb_], &constants)
                                .unwrap();
                            row_offset = squared.1;

                            // 除以 scale_factor
                            let squared = div_sf
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![squared.0.iter().collect()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = squared.1;

                            // 求和
                            let squared_sum = sum
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![squared.0.iter().collect()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = squared_sum.1;
                            squared_sum_features_emb.extend(squared_sum.0);
                        }

                        // 2.4 计算差值 (summed_features_emb_square - squared_sum_features_emb)
                        let diff = sub
                            .layout(
                                region,
                                row_offset,
                                &vec![
                                    summed_features_emb_square.iter().collect(),
                                    squared_sum_features_emb.iter().collect(),
                                ],
                                &constants,
                            )
                            .unwrap();
                        row_offset = diff.1;
                        let diff = diff.0;

                        // 新增：对diff做sum操作
                        let diff_sum = sum
                            .layout(region, row_offset, &vec![diff.iter().collect()], &constants)
                            .unwrap();
                        row_offset = diff_sum.1;
                        let diff_sum = diff_sum.0;

                        // 除以2
                        let half = div_same
                            .layout(
                                region,
                                row_offset,
                                &vec![diff_sum.iter().collect(), vec![two.as_ref()]],
                                &constants,
                            )
                            .unwrap();
                        row_offset = half.1;
                        let half = half.0;

                        // 3. 合并一阶和二阶特征
                        let final_output = add
                            .layout(
                                region,
                                row_offset,
                                &vec![sum_input.iter().collect(), half.iter().collect()],
                                &constants,
                            )
                            .unwrap();
                        row_offset = final_output.1;
                        outputs.extend(final_output.0);
                    }

                    Ok(outputs)
                },
            )
            .unwrap();

        Ok(vec![Array::from_shape_vec(
            IxDyn(&[input_shape[0], 1]),
            output.into_iter().map(|x| Rc::new(x)).collect(),
        )?])
    }

    fn backward(
        &self,
        mut _layouter: impl Layouter<F>,
        _inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        Ok(vec![])
    }
}

impl<F: PrimeField> NumericConsumer for FMChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::Sub,
            NumericType::DivSame,
            NumericType::DivSF,
            NumericType::Add,
            NumericType::Sum,
            NumericType::Square,
        ]
    }
}
