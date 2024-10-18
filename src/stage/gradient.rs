use std::{collections::BTreeSet, marker::PhantomData, rc::Rc};

use crate::{
    loss::{softmax::SoftMaxLossChip, Loss, LossType},
    numeric::{NumericConfig, NumericConsumer, NumericType},
    utils::{
        helpers::{to_field, FieldTensor, Tensor, NUMERIC_CONFIG},
        matcher::{match_configure, match_load_lookups},
        math::Int,
    },
};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    halo2curves::ff::PrimeField,
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};
use ndarray::{Array, ShapeError};

use super::assign::Assign;

#[derive(Clone, Debug)]
pub struct GradientCircuit<F: PrimeField> {
    pub score: Tensor,
    pub field_score: FieldTensor<F>,
    pub label: Vec<Int>,
    pub field_label: Vec<F>,
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
        let used_numerics = BTreeSet::from_iter(match loss {
            LossType::SoftMax => {
                Box::new(SoftMaxLossChip::<F>::construct(Default::default()))
                    as Box<dyn NumericConsumer>
            }
            .used_numerics(),
            _ => vec![],
        });
        let field_score = Array::from_shape_vec(
            score.shape(),
            score.iter().map(|x| to_field::<F>(*x)).collect(),
        )
        .unwrap();
        let field_label = label.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();

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
        let numeric_config = NUMERIC_CONFIG.lock().unwrap().clone();
        let loss_func = match self.loss {
            LossType::SoftMax => SoftMaxLossChip::<F>::compute,
            _ => panic!("Not implemented yet"),
        };
        loss_func(&self.score, &self.label, &numeric_config)
    }
}

impl<F: PrimeField> Assign<F> for GradientCircuit<F> {}

impl<F: PrimeField> Circuit<F> for GradientCircuit<F> {
    type Config = GradientConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        todo!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // Get numeric config from global state
        let numeric_config = NUMERIC_CONFIG.lock().unwrap().clone();

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
            .assign_vector(
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
                let softmax_loss_chip =
                    SoftMaxLossChip::<F>::construct(config.numeric_config.clone());
                softmax_loss_chip.compute(
                    layouter.namespace(|| "softmax loss"),
                    &assigned_score.view(),
                    &assigned_label,
                    &constants,
                )
            }
            _ => panic!("Not implemented yet"),
        }
        .unwrap();

        // Constrain the output
        for (i, cell) in output.iter().enumerate() {
            layouter
                .constrain_instance(cell.as_ref().cell(), config.public, i)
                .unwrap();
        }
        Ok(())
    }
}
