use std::{collections::BTreeSet, marker::PhantomData, rc::Rc, vec};

use crate::{
    loss::{softmax::SoftMaxLossChip, Loss, LossType},
    numeric::{NumericConfig, NumericConsumer, NumericType},
    utils::{
        helpers::{
            configure_static, get_circuit_numeric_config, get_numeric_config, to_field,
            FieldTensor, Tensor,
        },
        matcher::{match_configure, match_load_lookups},
        math::Int,
    },
};

use super::assign::Assign;
use crate::loss::sigmoid::SigmoidCrossEntropyLossChip;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    halo2curves::ff::PrimeField,
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};
use log::debug;
use ndarray::{Array, ShapeError};

#[derive(Clone, Debug, Default)]
pub struct GradientCircuit<F: PrimeField> {
    pub score: Tensor,
    pub field_score: FieldTensor<F>,
    pub label: Vec<Int>,
    pub field_label: FieldTensor<F>, // it need to be one-hot vectors
    pub loss: LossType,
    pub used_numerics: BTreeSet<NumericType>,
}

#[derive(Clone, Debug)]
pub struct GradientConfig<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> GradientCircuit<F> {
    pub fn construct(score: Tensor, label: Vec<Int>, loss: LossType) -> Self {
        assert_eq!(
            score.shape()[0],
            label.len(),
            "score and label batch size should be same"
        );
        let used_numerics = BTreeSet::from_iter(match loss {
            LossType::SoftMax => {
                Box::new(SoftMaxLossChip::<F>::construct(Default::default()))
                    as Box<dyn NumericConsumer>
            }
            .used_numerics(),
            LossType::Sigmoid => {
                Box::new(SigmoidCrossEntropyLossChip::<F>::construct(
                    Default::default(),
                )) as Box<dyn NumericConsumer>
            }
            .used_numerics(),
            _ => vec![],
        });
        let field_score = Array::from_shape_vec(
            score.shape(),
            score.iter().map(|x| to_field::<F>(*x)).collect(),
        )
        .unwrap();
        let one_hot = vec![0; score.shape()[1]];
        let field_label = label
            .iter()
            .map(|&x| {
                let mut one_hot = one_hot.clone();
                match loss {
                    LossType::SoftMax => {
                        one_hot[x as usize] = 1;
                        one_hot
                    }
                    LossType::Sigmoid => {
                        one_hot[0] = x;
                        one_hot
                    }
                    _ => panic!("Not implemented yet"),
                }
            })
            .flatten()
            .collect::<Vec<_>>();
        let field_label = Array::from_shape_vec(
            (label.len(), score.shape()[1]),
            field_label.iter().map(|x| to_field::<F>(*x)).collect(),
        )
        .unwrap()
        .into_dyn();

        Self {
            score,
            field_score,
            label,
            field_label,
            loss,
            used_numerics,
        }
    }

    pub fn run(&self) -> Result<(Int, Tensor), ShapeError> {
        let numeric_config = get_numeric_config();
        let loss_func = match self.loss {
            LossType::SoftMax => SoftMaxLossChip::<F>::compute,
            LossType::Sigmoid => SigmoidCrossEntropyLossChip::<F>::compute,
            _ => panic!("Not implemented yet"),
        };
        debug!("score:{}, label:{:?}", self.score, self.label);

        self.configure_numeric();

        loss_func(&self.score, &self.label, &numeric_config)
    }

    pub fn configure_numeric(&self) -> NumericConfig {
        configure_static(NumericConfig {
            batch_size: self.label.len(),
            used_numerics: self.used_numerics.clone().into(),
            ..get_numeric_config().clone()
        })
    }

    pub fn commitment_lengths(&self) -> Vec<usize> {
        vec![]
    }
}

impl<F: PrimeField> Assign<F> for GradientCircuit<F> {}

impl<F: PrimeField> Circuit<F> for GradientCircuit<F> {
    type Config = GradientConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Default::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // Get numeric config from global state
        let numeric_config = get_circuit_numeric_config(meta);

        // Create columns & constants
        let columns = (0..numeric_config.num_cols)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        for col in columns.iter() {
            meta.enable_equality(*col);
        }
        let constants = vec![meta.fixed_column()];
        for cst in constants.iter() {
            meta.enable_equality(*cst);
        }

        // Update numeric config
        let mut numeric_config = NumericConfig {
            columns,
            constants,
            ..numeric_config
        };
        // Add NaturalLookUp to the used numerics
        numeric_config
            .used_numerics
            .insert(NumericType::NaturalLookUp);

        // Configure each numerics
        let iter = <BTreeSet<NumericType> as Clone>::clone(&numeric_config.used_numerics.clone())
            .into_iter();
        for numeric_type in iter {
            numeric_config = match_configure(numeric_type)(meta, numeric_config);
        }

        // Create public column
        let public = meta.instance_column();
        meta.enable_equality(public);

        GradientConfig {
            numeric_config: Rc::new(numeric_config),
            public,
            _marker: PhantomData,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        // Assign tensors
        let assigned_score = self
            .assign_tensor(
                layouter.namespace(|| "assign_tensor"),
                &config.numeric_config.columns,
                &self.field_score,
            )
            .unwrap();
        let assigned_label = self
            .assign_tensor(
                layouter.namespace(|| "assign_label"),
                &config.numeric_config.columns,
                &self.field_label,
            )
            .unwrap();

        // Assign constants
        let constants = self
            .assign_constants(
                layouter.namespace(|| "assign_constants"),
                config.numeric_config.clone(),
            )
            .unwrap();

        // Load lookups
        for numeric_type in config.numeric_config.used_numerics.iter() {
            match match_load_lookups(
                config.numeric_config.clone(),
                *numeric_type,
                layouter.namespace(|| "load_lookups"),
            ) {
                Ok(_) => (),
                Err(e) => panic!(
                    "Error occurs at GradientCircuit.synthesize load lookups: {:?}",
                    e
                ),
            }
        }

        // Run the circuit by each operation chips
        let output = match self.loss {
            LossType::SoftMax => {
                let soft_max_loss_chip =
                    SoftMaxLossChip::<F>::construct(config.numeric_config.clone());
                soft_max_loss_chip.compute(
                    layouter.namespace(|| "softmax loss"),
                    &assigned_score.view(),
                    &assigned_label.view(),
                    &constants,
                )
            }
            LossType::Sigmoid => {
                let sigmoid_cross_entropy_loss_chip =
                    SigmoidCrossEntropyLossChip::<F>::construct(config.numeric_config.clone());
                sigmoid_cross_entropy_loss_chip.compute(
                    layouter.namespace(|| "sigmoid loss"),
                    &assigned_score.view(),
                    &assigned_label.view(),
                    &constants,
                )
            }
            _ => panic!("Not implemented yet"),
        }
        .unwrap();
        debug!("output: {:?}", output);

        // Constrain the output
        // for (i, cell) in output.iter().enumerate() {
        //     layouter
        //         .constrain_instance(cell.as_ref().cell(), config.public, i)
        //         .unwrap();
        // }
        Ok(())
    }
}
