use std::marker::PhantomData;

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    halo2curves::ff::{FromUniformBytes, PrimeField},
    plonk::{Circuit, ConstraintSystem, ErrorFront},
};
use ndarray::{Array, Dim, IxDyn};

#[derive(Clone, Debug, Default)]
pub struct FormatLayer<F: PrimeField> {
    pub layername: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub weight_shape: Vec<usize>,
    pub original_weights: Array<i64, Dim<[usize; 2]>>,
    pub field_weights: Array<F, IxDyn>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelCircuit<F: PrimeField> {
    pub k: usize,
    pub layers: Vec<FormatLayer<F>>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig<F: PrimeField> {
    //   pub gadget_config: Rc<GadgetConfig>,
    //   pub public_col: Column<Instance>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ModelCircuit<F> {
    pub fn construct(k: usize, layers: Vec<FormatLayer<F>>) -> Self {
        Self { k, layers }
    }
}

impl<F: PrimeField> Circuit<F> for ModelCircuit<F> {
    type Config = ModelConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        todo!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        todo!()
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        todo!()
    }
}
