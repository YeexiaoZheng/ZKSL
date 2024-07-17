use std::{
    collections::{BTreeSet, HashMap},
    marker::PhantomData,
    rc::Rc,
};

use crate::{
    commitments::poseidon::FixedPoseidonChip,
    graph::Graph,
    numerics::numeric::{NumericConfig, NumericType},
    operations::{
        gemm::GemmChip,
        none::NoneChip,
        operation::{OPType, Operation},
        relu::ReLUChip,
        softmax::SoftMaxChip,
    },
    utils::{
        helpers::{to_field, FieldTensor, Tensor, NUMERIC_CONFIG},
        matcher::{
            match_configure, match_consumer, match_forward, match_load_lookups, match_op_type,
        },
    },
    weight::AssignedWeight,
};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    halo2curves::ff::{FromUniformBytes, PrimeField},
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};
use ndarray::{Array, ShapeError};

use super::assign::Assign;

#[derive(Clone, Debug)]
pub struct ForwardCircuit<F: PrimeField + Ord + FromUniformBytes<64>> {
    pub graph: Graph,
    pub used_numerics: BTreeSet<NumericType>,
    pub field_tensor_map: HashMap<String, FieldTensor<F>>,
}

#[derive(Clone, Debug)]
pub struct ForwardConfig<F: PrimeField + Ord + FromUniformBytes<64>> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub hasher: Option<FixedPoseidonChip<F>>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField + Ord + FromUniformBytes<64>> ForwardCircuit<F> {
    pub fn construct(graph: Graph) -> Self {
        let mut used_numerics = BTreeSet::new();
        for node in graph.nodes.iter() {
            let op_type = match_op_type(node.op_type.clone());
            used_numerics.extend(match_consumer::<F>(op_type).used_numerics().iter())
        }

        let field_tensor_map = graph
            .tensor_map
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    Array::from_shape_vec(
                        v.shape(),
                        v.iter().map(|x| to_field::<F>(x.clone())).collect(),
                    )
                    .unwrap(),
                )
            })
            .collect();

        Self {
            graph,
            used_numerics,
            field_tensor_map,
        }
    }

    // pub fn load_input(&mut self, tensor: &Tensor) {
    //     self.graph
    //         .tensor_map
    //         .insert("input".to_string(), tensor.clone());
    //     let field_tensor = Array::from_shape_vec(
    //         tensor.shape(),
    //         tensor.iter().map(|x| to_field::<F>(x.clone())).collect(),
    //     )
    //     .unwrap();
    //     self.field_tensor_map
    //         .insert("input".to_string(), field_tensor);
    // }

    pub fn run(&mut self) -> Result<Tensor, ShapeError> {
        let numeric_config = NUMERIC_CONFIG.lock().unwrap().clone();

        for node in self.graph.nodes.iter() {
            let operation = match_forward::<F>(match_op_type(node.op_type.clone()));
            let outputs = operation(
                &node
                    .inputs
                    .iter()
                    .map(|x| {
                        match self.graph.tensor_map.get(x) {
                            Some(x) => x.clone(),
                            None => panic!(
                                "Error occurs at ForwardCircuit.run: tensor '{}' not found",
                                x
                            ),
                        }
                        .clone()
                    })
                    .collect::<Vec<Tensor>>(),
                &numeric_config,
                &node.attributes,
            )?;
            for (output_str, output) in node.outputs.iter().zip(outputs.into_iter()) {
                self.graph.tensor_map.insert(output_str.clone(), output);
            }
        }

        Ok(self.graph.tensor_map.get("output").unwrap().clone())
    }
}

impl<F: PrimeField + Ord + FromUniformBytes<64>> Assign<F> for ForwardCircuit<F> {}

impl<F: PrimeField + Ord + FromUniformBytes<64>> Circuit<F> for ForwardCircuit<F> {
    type Config = ForwardConfig<F>;
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

        // Create hasher
        let hasher = FixedPoseidonChip::configure(meta, numeric_config.clone());

        ForwardConfig {
            numeric_config: Rc::new(numeric_config),
            public,
            hasher,
            _marker: PhantomData,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        // Assign tensors
        let mut assigned_tensor_map = self
            .assign_tensor_map(
                layouter.namespace(|| "assign_tensor_map"),
                &config.numeric_config.columns,
                &self.field_tensor_map,
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
                    "Error occurs at ForwardCircuit.synthesize load lookups: {:?}",
                    e
                ),
            }
        }

        // Run the circuit by each operation chips
        for op in self.graph.nodes.iter() {
            print!("op: {:?}\t\t", op.op_type);
            // Get inputs
            let inputs = op
                .inputs
                .iter()
                .map(|x| assigned_tensor_map.get(x).unwrap().view())
                .collect();
            // Run the operation
            let outputs = match match_op_type(op.op_type.clone()) {
                OPType::GEMM => GemmChip::<F>::construct(config.numeric_config.clone()).forward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &op.attributes,
                ),
                OPType::ReLU => ReLUChip::<F>::construct(config.numeric_config.clone()).forward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &op.attributes,
                ),
                OPType::SoftMax => SoftMaxChip::<F>::construct(config.numeric_config.clone())
                    .forward(
                        layouter.namespace(|| op.op_type.clone()),
                        &inputs,
                        &constants,
                        &op.attributes,
                    ),
                OPType::None => NoneChip::<F>::construct(config.numeric_config.clone()).forward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &op.attributes,
                ),
                _ => panic!("Operation type not supported"),
            }
            .unwrap();
            // Insert the output to the assigned tensor map
            for (op, output) in op.outputs.iter().zip(outputs.into_iter()) {
                assigned_tensor_map.insert(op.clone(), output);
            }
            println!("forward circuit compute successfully!");
            // println!("{:?}", layouter.)
        }

        // Constrain the output
        let output = assigned_tensor_map.get("output").unwrap().clone();
        for (i, cell) in output.iter().enumerate() {
            layouter
                .constrain_instance(cell.as_ref().cell(), config.public, i)
                .unwrap();
        }

        // Hash the weights
        let mut hash_output = vec![];
        if config.hasher.is_some() {
            println!("hashing the weights...");
            let hasher = config.hasher.as_ref().unwrap();
            let weight = AssignedWeight::<F>::construct(
                self.graph.nodes.clone(),
                assigned_tensor_map.clone(),
            );
            hash_output.push(hasher.hash_vec_to_one(
                layouter.namespace(|| "hash_vec"),
                weight.to_vec(),
                constants[&0].clone(),
            )?);
        }

        // Constrain the hash output
        let offset = output.len();
        if config.hasher.is_some() {
            for (i, cell) in hash_output.iter().enumerate() {
                layouter
                    .constrain_instance(cell.cell(), config.public, i + offset)
                    .unwrap();
            }
        }
        Ok(())
    }
}
