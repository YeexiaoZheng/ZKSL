use std::{marker::PhantomData, mem};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    halo2curves::ff::PrimeField,
    plonk::{Circuit, ConstraintSystem, ErrorFront},
};
use ndarray::ShapeError;

use crate::numerics::numeric::{NumericConfig, NumericType, _NumericConfig};

use super::layer::{ConfigLayer, Layer, LayerConfig, NumericConsumer, Tensor};

#[derive(Clone, Debug, Default)]
pub struct FullyConnectedLayer<F: PrimeField> {
    pub config: LayerConfig<F>,
}

impl<F: PrimeField> FullyConnectedLayer<F> {
    pub fn construct(config: LayerConfig<F>) -> Self {
        Self { config }
    }
}

impl<F: PrimeField> ConfigLayer<F> for FullyConnectedLayer<F> {
    fn config(&self) -> &LayerConfig<F> {
        &self.config
    }

    fn forward(&self, input: Tensor) -> Result<Tensor, ShapeError> {
        assert_eq!(input.ndim(), 2);
        assert_eq!(input.ndim(), self.config.input_shape.len());
        assert_eq!(self.config.input_shape[1], self.config.weight_shape[0]);
        assert_eq!(self.config.weight_shape[1], self.config.output_shape[0]);

        let input_shape = (self.config.input_shape[0], self.config.input_shape[1]);
        let input = input.into_shape(input_shape)?;
        let weight_shape = (self.config.weight_shape[0], self.config.weight_shape[1]);
        let weight = self.config.o_weight.clone().into_shape(weight_shape)?;

        Ok(input.dot(&weight).into_dyn())
    }
}

#[derive(Clone, Debug, Default)]
pub struct FullyConnectedChip<F: PrimeField> {
    pub config: LayerConfig<F>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> Layer<F> for FullyConnectedChip<F> {
    fn _forward(&self, input: Tensor) -> Result<Tensor, ShapeError> {
        Ok(FullyConnectedLayer::construct(self.config.clone()).forward(input)?)
    }

    fn forward(&self) {
        todo!()
    }
}

impl<F: PrimeField> NumericConsumer for FullyConnectedChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        let mut numerics = vec![];
        numerics.push(NumericType::Dot);
        numerics
    }
}

pub struct FullyConnectedCircuit<F: PrimeField> {
    input: Vec<F>,
    output: Vec<F>,
    weight: Vec<F>,
}

impl<F: PrimeField> FullyConnectedCircuit<F> {
    pub fn construct(config: LayerConfig<F>) -> Self {
        Self {
            input: vec![],
            output: vec![],
            weight: vec![],
        }
    }
}

impl<F: PrimeField> Circuit<F> for FullyConnectedCircuit<F> {
    type Config = _NumericConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        todo!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let k = 3;
        let columns = vec![meta.advice_column()];
        for col in columns.iter() {
            meta.enable_equality(*col);
        }
        let fixed = vec![meta.fixed_column()];
        for fix in fixed.iter() {
            meta.enable_equality(*fix);
        }
        let public = meta.instance_column();
        meta.enable_equality(public);

        Self::Config {
            k,
            scale_factor: 1,
            num_rows: (1 << k),
            num_cols: 10,
            columns,
            fixed,
            public,
            use_selectors: true,
            selectors: todo!(),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        todo!()
    }
}
