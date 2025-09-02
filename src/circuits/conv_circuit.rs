use crate::utils::helpers::get_circuit_numeric_config;
use crate::{
    numeric::{NumericConfig, NumericType},
    operation::{conv::ConvChip, Operation},
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
pub struct ConvCircuitConfig<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

pub struct ConvCircuit<F: PrimeField> {
    pub input: FieldTensor<F>,
    pub weight: FieldTensor<F>,
    pub biases: FieldTensor<F>,
    pub inpgrad: FieldTensor<F>,
}

impl<F: PrimeField> ConvCircuit<F> {
    pub fn construct(
        input: FieldTensor<F>,
        weight: FieldTensor<F>,
        biases: FieldTensor<F>,
        inpgrad: FieldTensor<F>,
    ) -> Self {
        Self {
            input,
            weight,
            biases,
            inpgrad,
        }
    }

    pub fn forward(
        &self,
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        ConvChip::<F>::forward(inputs, numeric_config, attributes)
    }

    pub fn backward(
        &self,
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        ConvChip::<F>::backward(inputs, numeric_config, attributes)
    }
}

impl<F: PrimeField> Assign<F> for ConvCircuit<F> {}

impl<F: PrimeField> Circuit<F> for ConvCircuit<F> {
    type Config = ConvCircuitConfig<F>;
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

        numeric_config.used_numerics.insert(NumericType::DivSF);
        numeric_config.used_numerics.insert(NumericType::Add);
        numeric_config.used_numerics.insert(NumericType::Sum);
        numeric_config.used_numerics.insert(NumericType::DotVec);
        numeric_config.used_numerics.insert(NumericType::Update);

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
        let weight = self
            .assign_tensor(
                layouter.namespace(|| "assign_weight"),
                &config.numeric_config.columns,
                &self.weight,
            )
            .unwrap();
        let biases = self
            .assign_tensor(
                layouter.namespace(|| "assign_biases"),
                &config.numeric_config.columns,
                &self.biases,
            )
            .unwrap();
        let inpgrad = self
            .assign_tensor(
                layouter.namespace(|| "assign_inpgrad"),
                &config.numeric_config.columns,
                &self.inpgrad,
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

        let strides: Vec<f64> = vec![1.0, 1.0];
        let pads: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0];
        let mut b_tree_map = BTreeMap::new();
        b_tree_map.insert("strides".to_string(), strides);
        b_tree_map.insert("pads".to_string(), pads);

        let conv_chip = ConvChip::<F>::construct(config_rc.clone());

        // Forward pass
        let _outputs = conv_chip
            .forward(
                layouter.namespace(|| "conv forward"),
                &vec![input.view(), weight.view(), biases.view()],
                &constants,
                &random,
                &b_tree_map,
            )
            .unwrap();
        // println!("outputs: {:#?}", outputs);

        // Backward pass
        let outputs = conv_chip
            .backward(
                layouter.namespace(|| "conv backward"),
                &vec![inpgrad.view(), input.view(), weight.view()],
                &constants,
                &random,
                &b_tree_map,
            )
            .unwrap();
        // println!("conv_circuit, outputs: {:?}", outputs.len());
        // println!("conv_circuit, outputs: {:?}", outputs);

        // Constrain public output
        let mut public_layouter = layouter.namespace(|| "public");

        let _ = outputs.iter().map(|output| {
            for (i, cell) in output.iter().enumerate() {
                public_layouter
                    .constrain_instance(cell.cell(), config.public, i)
                    .unwrap();
            }
        });

        Ok(())
    }
}
