use std::fmt::Error;

use crate::layers::layer::LayerType;

fn layer_matcher(s: String) -> Result<LayerType, Error> {
    match s {
        _ => Err(Error),
    }
}