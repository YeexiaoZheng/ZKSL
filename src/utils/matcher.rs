use std::{collections::BTreeMap, rc::Rc};

use halo2_proofs::{
    circuit::Layouter,
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
};
use ndarray::ShapeError;

use super::helpers::Tensor;
use crate::operation::max_pool::MaxPoolChip;
use crate::operation::reshape::ReshapeChip;
use crate::{numeric::nonlinear::gather::GatherLookUp, operation::conv::ConvChip};
use crate::{
    numeric::{
        add::AddLayouter,
        add_same::AddSameLayouter,
        div::DivLayouter,
        div_same::DivSameLayouter,
        div_sf::DivSFLayouter,
        dot_vec::DotVecLayouter,
        max::MaxLayouter,
        mul::MulLayouter,
        mul_same::MulSameLayouter,
        nonlinear::{
            exp::ExpLookUp, ln::LnLookUp, natural::NaturalLookUp, relu::ReluLookUp,
            NonLinearNumericLayout,
        },
        square::SquareLayouter,
        sub::SubLayouter,
        sub_same::SubSameLayouter,
        sum::SumLayouter,
        update::UpdateLayouter,
        NumericConfig, NumericConsumer, NumericType,
    },
    operation::{
        add::AddChip, concat::ConcatChip, fm::FMChip, gather::GatherChip, gemm::GemmChip,
        mean::MeanChip, none::NoneChip, relu::ReLUChip, softmax::SoftMaxChip, squeeze::SqueezeChip,
        unsqueeze::UnsqueezeChip, OPType,
    },
};

pub fn match_op_type(op_type: String) -> OPType {
    match op_type.as_str() {
        "Concat" => OPType::Concat,
        "Unsqueeze" => OPType::Unsqueeze,
        "Squeeze" => OPType::Squeeze,
        "Add" => OPType::Add,

        "Gather" => OPType::Gather,
        "FM" => OPType::FM,

        "ReduceMean" => OPType::Mean,

        "Gemm" => OPType::GEMM,
        "MatMul" => OPType::GEMM,

        "Relu" => OPType::ReLU,
        "Softmax" => OPType::SoftMax,
        "Conv" => OPType::Conv,
        "MaxPool" => OPType::MaxPool,
        "Reshape" => OPType::Reshape,
        "None" => OPType::None,
        _ => OPType::None,
    }
}

pub fn match_forward<F: PrimeField>(
    op_type: OPType,
) -> fn(&Vec<Tensor>, &NumericConfig, &BTreeMap<String, Vec<f64>>) -> Result<Vec<Tensor>, ShapeError>
{
    match op_type {
        OPType::Concat => ConcatChip::<F>::forward,
        OPType::Squeeze => SqueezeChip::<F>::forward,
        OPType::Unsqueeze => UnsqueezeChip::<F>::forward,
        OPType::Gather => GatherChip::<F>::forward,
        OPType::FM => FMChip::<F>::forward,
        OPType::Mean => MeanChip::<F>::forward,
        OPType::GEMM => GemmChip::<F>::forward,
        OPType::Add => AddChip::<F>::forward,
        OPType::ReLU => ReLUChip::<F>::forward,
        OPType::SoftMax => SoftMaxChip::<F>::forward,
        OPType::Conv => ConvChip::<F>::forward,
        OPType::MaxPool => MaxPoolChip::<F>::forward,
        OPType::Reshape => ReshapeChip::<F>::forward,
        _ => NoneChip::<F>::forward,
    }
}

pub fn match_backward<F: PrimeField>(
    op_type: OPType,
) -> fn(&Vec<Tensor>, &NumericConfig, &BTreeMap<String, Vec<f64>>) -> Result<Vec<Tensor>, ShapeError>
{
    match op_type {
        OPType::Concat => ConcatChip::<F>::backward,
        OPType::Squeeze => SqueezeChip::<F>::backward,
        OPType::Unsqueeze => UnsqueezeChip::<F>::backward,
        OPType::Gather => GatherChip::<F>::backward,
        OPType::FM => FMChip::<F>::backward,
        OPType::Mean => MeanChip::<F>::backward,
        OPType::GEMM => GemmChip::<F>::backward,
        OPType::Add => AddChip::<F>::backward,
        OPType::ReLU => ReLUChip::<F>::backward,
        OPType::SoftMax => SoftMaxChip::<F>::backward,
        OPType::Conv => ConvChip::<F>::backward,
        OPType::MaxPool => MaxPoolChip::<F>::backward,
        OPType::Reshape => ReshapeChip::<F>::backward,
        _ => NoneChip::<F>::backward,
    }
}

pub fn match_configure<F: PrimeField>(
    numeric_type: NumericType,
) -> fn(&mut ConstraintSystem<F>, NumericConfig) -> NumericConfig {
    match numeric_type {
        NumericType::Square => SquareLayouter::<F>::configure,
        NumericType::Max => MaxLayouter::<F>::configure,
        NumericType::Sum => SumLayouter::<F>::configure,
        NumericType::Add => AddLayouter::<F>::configure,
        NumericType::AddSame => AddSameLayouter::<F>::configure,
        NumericType::Sub => SubLayouter::<F>::configure,
        NumericType::SubSame => SubSameLayouter::<F>::configure,
        NumericType::Mul => MulLayouter::<F>::configure,
        NumericType::MulSame => MulSameLayouter::<F>::configure,
        NumericType::Div => DivLayouter::<F>::configure,
        NumericType::DivSame => DivSameLayouter::<F>::configure,
        NumericType::DivSF => DivSFLayouter::<F>::configure,
        NumericType::DotVec => DotVecLayouter::<F>::configure,
        NumericType::Update => UpdateLayouter::<F>::configure,

        NumericType::NaturalLookUp => NaturalLookUp::<F>::configure,
        NumericType::Relu => ReluLookUp::<F>::configure,
        NumericType::Exp => ExpLookUp::<F>::configure,
        NumericType::Ln => LnLookUp::<F>::configure,
        NumericType::Gather => GatherLookUp::<F>::configure,
    }
}

pub fn match_consumer<F: PrimeField>(op_type: OPType) -> Box<dyn NumericConsumer> {
    match op_type {
        OPType::Concat => {
            Box::new(ConcatChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::Unsqueeze => {
            Box::new(UnsqueezeChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::Squeeze => {
            Box::new(SqueezeChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::Gather => {
            Box::new(GatherChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::FM => {
            Box::new(FMChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::Mean => {
            Box::new(MeanChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::GEMM => {
            Box::new(GemmChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::Add => {
            Box::new(AddChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::ReLU => {
            Box::new(ReLUChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::SoftMax => {
            Box::new(SoftMaxChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::Conv => {
            Box::new(ConvChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::MaxPool => {
            Box::new(MaxPoolChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
        }
        OPType::Reshape => {
            Box::new(ReshapeChip::<F>::construct(Default::default())) as Box<dyn NumericConsumer>
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
        // Input lookup always has to be loaded(natural numbers)
        NumericType::NaturalLookUp => NaturalLookUp::<F>::construct(numeric_config)
            .load_lookups(layouter.namespace(|| "natural lookup")),

        // Non-linear lookups
        NumericType::Relu => ReluLookUp::<F>::construct(numeric_config)
            .load_lookups(layouter.namespace(|| "relu lookup")),
        NumericType::Exp => ExpLookUp::<F>::construct(numeric_config)
            .load_lookups(layouter.namespace(|| "exp lookup")),
        NumericType::Ln => LnLookUp::<F>::construct(numeric_config)
            .load_lookups(layouter.namespace(|| "ln lookup")),

        _ => Ok(()),
    }
}
