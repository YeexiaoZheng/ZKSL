use std::{
    collections::{BTreeSet, HashMap},
    marker::PhantomData,
    rc::Rc,
};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    halo2curves::ff::PrimeField,
    plonk::{Circuit, Column, ConstraintSystem, ErrorFront, Instance},
};
use ndarray::ShapeError;

use crate::{
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
        helpers::{FieldTensor, Tensor, NUMERIC_CONFIG},
        matcher::{match_backward, match_configure, match_load_lookups, match_op_type},
    },
};

use super::initialize::Initialize;

#[derive(Clone, Debug)]
pub struct BackwardCircuit<F: PrimeField> {
    pub graph: Graph,
    pub used_numerics: BTreeSet<NumericType>,
    pub field_tensor_map: HashMap<String, FieldTensor<F>>,
}

#[derive(Clone, Debug)]
pub struct BackwardConfig<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> Initialize<F> for BackwardCircuit<F> {
    fn construct(graph: Graph) -> Self {
        let (used_numerics, field_tensor_map) = Self::initialize(graph.clone());
        Self {
            graph,
            used_numerics,
            field_tensor_map,
        }
    }

    fn run(&self, _tensor: &Tensor) -> Result<Tensor, ShapeError> {
        let mut tensor_map = self.graph.tensor_map.clone();
        let numeric_config = NUMERIC_CONFIG.lock().unwrap().clone();

        for node in self.graph.nodes.iter() {
            let operation = match_backward::<F>(match_op_type(node.op_type.clone()));
            let outputs = operation(
                &node
                    .inputs
                    .iter()
                    .map(|x| tensor_map.get(x).unwrap().clone())
                    .collect::<Vec<Tensor>>(),
                &numeric_config,
                &node.attributes,
            )?;
            for (op, output) in node.outputs.iter().zip(outputs.into_iter()) {
                tensor_map.insert(op.clone(), output);
            }
        }

        Ok(tensor_map.get("output").unwrap().clone())
    }
}

impl<F: PrimeField> Circuit<F> for BackwardCircuit<F> {
    type Config = BackwardConfig<F>;
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

        BackwardConfig {
            numeric_config: Rc::new(numeric_config),
            public,
            _marker: PhantomData,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
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
                    "Error occurs at BackwardCircuit.synthesize load lookups: {:?}",
                    e
                ),
            }
        }

        // Run the circuit by each operation chips
        for op in self.graph.nodes.iter() {
            // Get inputs
            let inputs = op
                .inputs
                .iter()
                .map(|x| assigned_tensor_map.get(x).unwrap().view())
                .collect();
            // Run the operation
            let outputs = match match_op_type(op.op_type.clone()) {
                OPType::GEMM => GemmChip::<F>::construct(config.numeric_config.clone()).backward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &op.attributes,
                ),
                OPType::ReLU => ReLUChip::<F>::construct(config.numeric_config.clone()).backward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &op.attributes,
                ),
                OPType::SoftMax => SoftMaxChip::<F>::construct(config.numeric_config.clone())
                    .backward(
                        layouter.namespace(|| op.op_type.clone()),
                        &inputs,
                        &constants,
                        &op.attributes,
                    ),
                OPType::None => NoneChip::<F>::construct(config.numeric_config.clone()).backward(
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
        }

        // Constrain the output
        let output = assigned_tensor_map.get("output").unwrap().clone();
        for (i, cell) in output.iter().enumerate() {
            layouter
                .constrain_instance(cell.as_ref().cell(), config.public, i)
                .unwrap();
        }
        Ok(())
    }
}
