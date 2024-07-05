use std::{collections::HashMap, marker::PhantomData, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, Axis, ShapeError};

use crate::{
    numerics::{accumulator::AccumulatorChip, div::{self, DivChip}, mul::MulChip, nonlinear::{exp::{self, ExpChip}, ln::LnChip}, numeric::NumericConfig, sub::{self, SubChip}},
    utils::{
        helpers::{AssignedTensorRef, CellRc, Tensor},
        math::{exp, ln},
    },
};

use super::loss::Loss;

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
        label: &Vec<i64>,
        numeric_config: &NumericConfig,
    ) -> Result<(i64, Tensor), ShapeError> {
        // Check input shape
        assert_eq!(input.ndim(), 2);
        assert_eq!(input.shape()[0], label.len());

        let n = input.shape()[0] as i64;
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

        let l: i64 = f
            .outer_iter()
            .zip(label.iter())
            .map(|(row, &y)| row[y as usize])
            .sum();
        let loss = -l + ln(efs.mapv(|x| ln(x, numeric_config.scale_factor)).sum(), numeric_config.scale_factor);

        let efs = Array::from_shape_vec(
            input.shape(),
            efs.iter()
                .map(|&x| vec![x; input.shape()[1]])
                .flatten()
                .collect::<Vec<_>>(),
        )?;
        let mut dscore = ef / efs;
        for (i, y) in label.iter().enumerate() {
            dscore[[i, *y as usize]] -= numeric_config.scale_factor as i64;
        }
        Ok((loss / n, dscore / n))
    }
}

impl<F: PrimeField> Loss<F> for SoftMaxLossChip<F> {
    fn compute(
        &self,
        layouter: impl Layouter<F>,
        inputs: &AssignedTensorRef<F>,
        label: &Vec<CellRc<F>>,
        constants: &HashMap<i64, CellRc<F>>,
    ) -> () {
        let exp_chip = ExpChip::<F>::construct(self.numeric_config.clone());
        let ln_chip = LnChip::<F>::construct(self.numeric_config.clone());
        let sub_chip = SubChip::<F>::construct(self.numeric_config.clone());
        let mul_chip = MulChip::<F>::construct(self.numeric_config.clone());
        let div_chip = DivChip::<F>::construct(self.numeric_config.clone());
        let acc_chip = AccumulatorChip::<F>::construct(self.numeric_config.clone());
    }
}
