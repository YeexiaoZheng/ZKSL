use std::{collections::HashMap, marker::PhantomData, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, Axis, ShapeError};

use crate::{
    numeric::{
        accumulator::AccumulatorChip, div::DivChip, max::MaxChip, mul::MulChip,
        nonlinear::exp::ExpChip, sub::SubChip, Numeric, NumericConfig, NumericConsumer,
        NumericType,
    },
    utils::{
        helpers::{to_primitive, AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{exp, ln, Int},
    },
};

use super::Loss;

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
                .axis_iter(Axis(0))
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
        label: &Vec<CellRc<F>>,
        constants: &HashMap<Int, CellRc<F>>,
    ) -> Result<AssignedTensor<F>, ShapeError> {
        let exp_chip = ExpChip::<F>::construct(self.numeric_config.clone());
        let max_chip = MaxChip::<F>::construct(self.numeric_config.clone());
        let sub_chip = SubChip::<F>::construct(self.numeric_config.clone());
        let mul_chip = MulChip::<F>::construct(self.numeric_config.clone());
        let div_chip = DivChip::<F>::construct(self.numeric_config.clone());
        let acc_chip = AccumulatorChip::<F>::construct(self.numeric_config.clone());

        let zero = constants.get(&0).unwrap().clone();
        let one = constants.get(&1).unwrap().clone();
        let sf = constants
            .get(&(self.numeric_config.scale_factor as Int))
            .unwrap()
            .clone();
        let bs = constants
            .get(&(self.numeric_config.batch_size as Int))
            .unwrap()
            .clone();

        let mut f = vec![];
        for row in input.outer_iter() {
            let row = row.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
            let max = match max_chip.compute(
                layouter.namespace(|| "SoftMaxLoss Max compute"),
                &vec![row.clone()],
                &vec![],
            ) {
                Ok(output) => output[0].clone(),
                Err(_) => panic!("SoftMaxLoss max compute failed"),
            };
            f.extend(
                match sub_chip.compute(
                    layouter.namespace(|| "SoftMaxLoss sub compute"),
                    &vec![row.clone(), vec![&max; row.len()]],
                    &vec![zero.as_ref()],
                ) {
                    Ok(output) => output,
                    Err(_) => panic!("SoftMaxLoss sub compute failed"),
                },
            )
        }

        let ef = match exp_chip.compute(
            layouter.namespace(|| "SoftMaxLoss Exp compute"),
            &vec![f.iter().collect()],
            &vec![zero.as_ref(), one.as_ref()],
        ) {
            Ok(output) => output,
            Err(_) => panic!("SoftMaxLoss exp compute failed"),
        };

        let ef = Array::from_shape_vec(input.shape(), ef)?;
        let mut dscore = vec![];
        for row in ef.outer_iter() {
            let sum = match acc_chip.compute(
                layouter.namespace(|| "SoftMax Loss Accumulator compute"),
                &vec![row.iter().collect()],
                &vec![zero.as_ref()],
            ) {
                Ok(output) => output[0].clone(),
                Err(_) => panic!("SoftMax Loss accumulator compute failed"),
            };
            // Multiply by scale factor before division
            let row = match mul_chip.compute(
                layouter.namespace(|| "SoftMax Loss Mul compute"),
                &vec![row.iter().collect(), vec![sf.as_ref(); row.len()]],
                &vec![zero.as_ref()],
            ) {
                Ok(output) => output,
                Err(_) => panic!("SoftMax Loss mul compute failed"),
            };
            dscore.extend(
                match div_chip.compute(
                    layouter.namespace(|| "SoftMax Loss Div compute"),
                    &vec![row.iter().collect(), vec![&sum; row.len()]],
                    &vec![zero.as_ref(), one.as_ref()],
                ) {
                    Ok(output) => output,
                    Err(_) => panic!("SoftMax Loss div compute failed"),
                },
            )
        }
        let mut dscore = Array::from_shape_vec(input.shape(), dscore)?;

        let mut dscore_label_view = vec![];
        for (i, y) in label.iter().enumerate() {
            y.value()
                .map(|y| dscore_label_view.push(&dscore[[i, to_primitive::<F>(y) as usize]]));
        }

        let dscore_sub = match sub_chip.compute(
            layouter.namespace(|| "SoftMax Loss Sub compute"),
            &vec![
                dscore_label_view.clone(),
                vec![sf.as_ref(); dscore_label_view.len()],
            ],
            &vec![zero.as_ref()],
        ) {
            Ok(output) => output,
            Err(_) => panic!("SoftMax Loss sub compute failed"),
        };

        for (i, y) in label.iter().enumerate() {
            y.value()
                .map(|y| dscore[[i, to_primitive::<F>(y) as usize]] = dscore_sub[i].clone());
        }

        let dscore = match div_chip.compute(
            layouter.namespace(|| "SoftMax Loss Div compute"),
            &vec![dscore.iter().collect(), vec![bs.as_ref(); dscore.len()]],
            &vec![zero.as_ref(), one.as_ref()],
        ) {
            Ok(output) => output,
            Err(_) => panic!("SoftMax Loss div compute failed"),
        };

        Ok(Array::from_shape_vec(
            input.shape(),
            dscore.into_iter().map(|x| Rc::new(x)).collect(),
        )?)
    }
}

impl<F: PrimeField> NumericConsumer for SoftMaxLossChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::RowLookUp,
            NumericType::Exp,
            NumericType::Max,
            NumericType::Sub,
            NumericType::Mul,
            NumericType::Div,
            NumericType::Accumulator,
        ]
    }
}
