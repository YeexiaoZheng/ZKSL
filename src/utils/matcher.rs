use std::marker::PhantomData;

use halo2_proofs::halo2curves::ff::PrimeField;

use crate::layers::{fully_connected::FullyConnectedChip, layer::{LayerConfig, LayerType, NumericConsumer}, none::NoneChip};

pub fn match_layer_name_to_layer_type(layer_name: String) -> LayerType {
    match layer_name.as_str() {
        "FullyConnected" => LayerType::FullyConnected,
        "ReLU" => LayerType::ReLU,
        "None" => LayerType::None,
        _ => LayerType::None,
    }
}

pub fn match_layer_type_to_consumer<F: PrimeField>(layer_type: LayerType, layer_config: LayerConfig<F>) -> Box<dyn NumericConsumer> {
    match layer_type {
        LayerType::FullyConnected => Box::new(FullyConnectedChip {
            config: layer_config,
            _marker: PhantomData::<F>,
        }) as Box<dyn NumericConsumer>,
        _ => Box::new(NoneChip {}) as Box<dyn NumericConsumer>,
    }
}