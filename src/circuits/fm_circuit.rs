use crate::utils::helpers::get_circuit_numeric_config;
use crate::{
    numeric::{NumericConfig, NumericType},
    operation::{fm::FMChip, Operation},
    stage::assign::Assign,
    utils::{
        helpers::{FieldTensor, Tensor},
        matcher::{match_configure, match_load_lookups},
    },
};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    halo2curves::ff::PrimeField,
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};
use ndarray::ShapeError;
use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    rc::Rc,
};

#[derive(Clone, Debug)]
pub struct FMConfig<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

pub struct FMCircuit<F: PrimeField> {
    pub input: FieldTensor<F>,
    pub embedding: FieldTensor<F>,
    pub label: Vec<F>,
}

impl<F: PrimeField> FMCircuit<F> {
    pub fn construct(input: FieldTensor<F>, embedding: FieldTensor<F>, label: Vec<F>) -> Self {
        Self {
            input,
            embedding,
            label,
        }
    }

    pub fn forward(
        &self,
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        FMChip::<F>::forward(inputs, numeric_config, _attributes)
        // FMChipPrimitive::<F>::forward(inputs, numeric_config, _attributes)
    }
}

impl<F: PrimeField> Assign<F> for FMCircuit<F> {}

impl<F: PrimeField> Circuit<F> for FMCircuit<F> {
    type Config = FMConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        todo!()
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

        numeric_config.used_numerics.insert(NumericType::Sub);
        numeric_config.used_numerics.insert(NumericType::DivSame);
        numeric_config.used_numerics.insert(NumericType::DivSF);
        numeric_config.used_numerics.insert(NumericType::Div);
        numeric_config.used_numerics.insert(NumericType::Add);
        numeric_config.used_numerics.insert(NumericType::Sum);
        numeric_config.used_numerics.insert(NumericType::Square);

        // Configure each numerics
        let iter = <BTreeSet<NumericType> as Clone>::clone(&numeric_config.used_numerics.clone())
            .into_iter();
        for numeric_type in iter {
            numeric_config = match_configure(numeric_type)(meta, numeric_config);
        }

        // Create public column
        let public = meta.instance_column();
        meta.enable_equality(public);

        Self::Config {
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
        // Construct Exp chip
        let config_rc = config.numeric_config.clone();

        // Assign input tensors
        let input = self
            .assign_tensor(
                layouter.namespace(|| "assign_input"),
                &config.numeric_config.columns,
                &self.input,
            )
            .unwrap();
        let embedding = self
            .assign_tensor(
                layouter.namespace(|| "assign_embedding"),
                &config.numeric_config.columns,
                &self.embedding,
            )
            .unwrap();

        // Assign constants
        let constants = self
            .assign_constants(layouter.namespace(|| "assign_constants"), config_rc.clone())
            .unwrap();

        // Assign random
        let random = self
            .assign_random(
                layouter.namespace(|| "assign_random"),
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
                    "Error occurs at ForwardCircuit.synthesize load lookups: {:?}",
                    e
                ),
            }
        }

        let fm_chip = FMChip::<F>::construct(config_rc.clone());
        // let fm_chip = FMChipPrimitive::<F>::construct(config_rc.clone());

        // Forward pass
        let outputs = fm_chip
            .forward(
                layouter.namespace(|| "fm forward"),
                &vec![input.view(), embedding.view()],
                &constants,
                &random,
                &BTreeMap::new(),
            )
            .unwrap();
        // println!("outputs: {:#?}", outputs);
        // println!("public: {:#?}", config.public);

        // Constrain public output
        let mut public_layouter = layouter.namespace(|| "public");
        for (i, cell) in outputs[0].iter().enumerate() {
            public_layouter.constrain_instance(cell.cell(), config.public, i)?;
        }

        Ok(())
    }
}
