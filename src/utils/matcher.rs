use std::collections::HashMap;

use halo2_proofs::{halo2curves::ff::PrimeField, plonk::ConstraintSystem};
use ndarray::ShapeError;

use crate::{
    numerics::{
        accumulator::AccumulatorChip,
        dot::DotChip,
        numeric::{NumericConfig, NumericType},
    },
    operations::{
        gemm::GemmChip,
        none::NoneChip,
        operation::{NumericConsumer, OPType},
    },
};

use super::helpers::Tensor;

pub fn match_op_type(op_type: String) -> OPType {
    match op_type.as_str() {
        "Gemm" => OPType::GEMM,
        "ReLU" => OPType::ReLU,
        "None" => OPType::None,
        _ => OPType::None,
    }
}

pub fn match_operation<F: PrimeField>(
    op_type: OPType,
) -> fn(&Vec<Tensor>, &HashMap<String, f64>) -> Result<Vec<Tensor>, ShapeError> {
    match op_type {
        OPType::GEMM => GemmChip::<F>::forward,
        _ => NoneChip::<F>::forward,
    }
}

pub fn match_configure<F: PrimeField>(
    numeric_type: NumericType,
) -> fn(&mut ConstraintSystem<F>, NumericConfig) -> NumericConfig {
    match numeric_type {
        NumericType::Dot => DotChip::<F>::configure,
        NumericType::Accumulator => AccumulatorChip::<F>::configure,
    }
}

pub fn match_consumer<F: PrimeField>(op_type: OPType) -> Box<dyn NumericConsumer> {
    match op_type {
        OPType::GEMM => {
            Box::new(GemmChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        _ => Box::new(NoneChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>,
    }
}
