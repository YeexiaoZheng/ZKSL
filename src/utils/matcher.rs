use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
    circuit::Layouter,
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
};
use ndarray::ShapeError;

use crate::{
    numerics::{
        accumulator::AccumulatorChip, add::AddChip, div::DivChip, dot::DotChip, lookups::{field_lookup::FieldLookUpChip, row_lookup::RowLookUpChip}, mul::MulChip, nonlinear::{exp::ExpChip, ln::LnChip, nonlinear::NonLinearNumeric, relu::ReluChip}, numeric::{NumericConfig, NumericType}, sub::SubChip
    },
    operations::{
        gemm::GemmChip,
        none::NoneChip,
        operation::{NumericConsumer, OPType},
        relu::ReLUChip,
    },
};

use super::helpers::Tensor;

pub fn match_op_type(op_type: String) -> OPType {
    match op_type.as_str() {
        "Gemm" => OPType::GEMM,
        "ReLU" => OPType::ReLU,
        "SoftMax" => OPType::SoftMax,
        "None" => OPType::None,
        _ => OPType::None,
    }
}

pub fn match_operation<F: PrimeField>(
    op_type: OPType,
) -> fn(&Vec<Tensor>, &NumericConfig, &HashMap<String, f64>) -> Result<Vec<Tensor>, ShapeError> {
    match op_type {
        OPType::GEMM => GemmChip::<F>::forward,
        OPType::ReLU => ReLUChip::<F>::forward,
        _ => NoneChip::<F>::forward,
    }
}

pub fn match_configure<F: PrimeField>(
    numeric_type: NumericType,
) -> fn(&mut ConstraintSystem<F>, NumericConfig) -> NumericConfig {
    match numeric_type {
        NumericType::RowLookUp => RowLookUpChip::<F>::configure,
        NumericType::FieldLookUp => FieldLookUpChip::<F>::configure,
        NumericType::Add => AddChip::<F>::configure,
        NumericType::Sub => SubChip::<F>::configure,
        NumericType::Mul => MulChip::<F>::configure,
        NumericType::Div => DivChip::<F>::configure,
        NumericType::Dot => DotChip::<F>::configure,
        NumericType::Accumulator => AccumulatorChip::<F>::configure,
        NumericType::Relu => ReluChip::<F>::configure,
        NumericType::Exp => ExpChip::<F>::configure,
        NumericType::Ln => LnChip::<F>::configure,
    }
}

pub fn match_consumer<F: PrimeField>(op_type: OPType) -> Box<dyn NumericConsumer> {
    match op_type {
        OPType::GEMM => {
            Box::new(GemmChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::ReLU => {
            Box::new(ReLUChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::SoftMax => {
            Box::new(NoneChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        _ => Box::new(NoneChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>,
    }
}

pub fn match_load_lookups<F: PrimeField>(
    numeric_config: Rc<NumericConfig>,
    numeric_type: NumericType,
    mut layouter: impl Layouter<F>,
) -> Result<(), Error> {
    match numeric_type {
        // Input lookups
        NumericType::RowLookUp => RowLookUpChip::<F>::construct(numeric_config)
            .load_lookups(layouter.namespace(|| "row lookup")),
        NumericType::FieldLookUp => FieldLookUpChip::<F>::construct(numeric_config)
            .load_lookups(layouter.namespace(|| "field lookup")),

        // Non-linear lookups
        NumericType::Relu => ReluChip::<F>::construct(numeric_config)
            .load_lookups(layouter.namespace(|| "relu lookup")),
        NumericType::Exp => ExpChip::<F>::construct(numeric_config)
            .load_lookups(layouter.namespace(|| "exp lookup")),
        _ => Ok(()),
    }
}
