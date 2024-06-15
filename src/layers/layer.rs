use halo2_proofs::{circuit::AssignedCell, halo2curves::ff::PrimeField};
use ndarray::{Array, IxDyn, ShapeError};

use crate::{
    model::FormatLayer, numerics::numeric::NumericType,
    utils::matcher::match_layer_name_to_layer_type,
};

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum LayerType {
    FullyConnected,
    ReLU,
    #[default]
    None,
}

pub type Tensor = Array<i64, IxDyn>;
pub type CellRc<F> = AssignedCell<F, F>;
pub type AssignedTensor<F> = Array<CellRc<F>, IxDyn>;

#[derive(Clone, Debug, Default)]
pub struct LayerConfig<F: PrimeField> {
    pub layer_type: LayerType,
    pub input_shape: Vec<usize>,
    // pub input_configs: Vec<&'a LayerConfig<'a, F>>,
    pub output_shape: Vec<usize>,
    // pub output_configs: Vec<&'a LayerConfig<'a, F>>,
    pub weight_shape: Vec<usize>,
    pub o_weight: Tensor,
    pub f_weight: Array<F, IxDyn>,
    pub params: Vec<i64>, // This is turned into layer specific configurations at runtime
    pub mask: Vec<i64>,
}

impl<F: PrimeField> LayerConfig<F> {
    pub fn construct(layer: FormatLayer<F>) -> Self {
        Self {
            layer_type: match_layer_name_to_layer_type(layer.layer_name),
            input_shape: layer.input_shape,
            output_shape: layer.output_shape,
            weight_shape: layer.weight_shape,
            o_weight: layer.original_weights.into_dyn(),
            f_weight: layer.field_weights,
            params: vec![],
            mask: vec![],
        }
    }
}

pub trait ConfigLayer<F: PrimeField> {
    fn config(&self) -> &LayerConfig<F>;
    fn forward(&self, input: Tensor) -> Result<Tensor, ShapeError>;
}

pub trait Layer<F: PrimeField> {
    fn _forward(&self, input: Tensor) -> Result<Tensor, ShapeError>;

    fn forward(&self);
}

pub trait NumericConsumer {
    fn used_numerics(&self) -> Vec<NumericType>;
}
