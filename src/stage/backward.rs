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
use ndarray::{Array, ShapeError};

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
        helpers::{to_field, FieldTensor, Tensor, NUMERIC_CONFIG},
        matcher::{
            match_backward, match_configure, match_consumer, match_load_lookups, match_op_type,
        },
        math::Int,
    },
};

use super::assign::Assign;

#[derive(Clone, Debug)]
pub struct BackwardCircuit<F: PrimeField> {
    pub graph: Graph,
    pub used_numerics: BTreeSet<NumericType>,
    pub field_tensor_map: HashMap<String, FieldTensor<F>>,
    pub lr: Int,
    pub field_lr: F,
}

#[derive(Clone, Debug)]
pub struct BackwardConfig<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> BackwardCircuit<F> {
    pub fn construct(graph: Graph, lr: Int) -> Self {
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
            lr,
            field_lr: to_field::<F>(lr),
        }
    }

    pub fn run(&mut self) -> Result<Tensor, ShapeError> {
        let mut res = self.graph.tensor_map.get("gradient").unwrap().clone();
        let numeric_config = NUMERIC_CONFIG.lock().unwrap().clone();

        for node in self.graph.nodes.iter().rev() {
            let operation = match_backward::<F>(match_op_type(node.op_type.clone()));
            // println!(
            //     "Running operation: {}, backward inputs: {:?}",
            //     node.op_type, node.backward_inputs
            // );
            let outputs = operation(
                &node
                    .backward_inputs
                    .iter()
                    .map(|x| match self.graph.tensor_map.get(x) {
                        Some(x) => x.clone(),
                        None => panic!("Tensor not found: {}", x),
                    })
                    .collect::<Vec<Tensor>>(),
                &numeric_config,
                &node.attributes,
            )?;
            res = outputs[0].clone();
            for (output_str, output) in node.backward_outputs.iter().zip(outputs.into_iter()) {
                // Update the weight
                if output_str.contains(".grad") {
                    let k = output_str
                        .clone()
                        .strip_suffix(".grad")
                        .unwrap()
                        .to_string();
                    let weight = self.graph.tensor_map.get(&k).unwrap();
                    let new_weight = weight.clone() - output.clone() / self.lr;
                    *self.graph.tensor_map.entry(k).or_default() = new_weight;
                }
                self.graph.tensor_map.insert(output_str.clone(), output);
            }
        }

        Ok(res)
    }
}

impl<F: PrimeField> Assign<F> for BackwardCircuit<F> {}

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
        let mut res = assigned_tensor_map.get("gradient").unwrap().clone();
        for op in self.graph.nodes.iter().rev() {
            // Get inputs
            let inputs = op
                .backward_inputs
                .iter()
                .map(|x| {
                    match assigned_tensor_map.get(x) {
                        Some(x) => x,
                        None => panic!("Tensor not found: {}", x),
                    }
                    .view()
                })
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
            res = outputs[0].clone();
            // Insert the output to the assigned tensor map
            for (output_str, output) in op.backward_outputs.iter().zip(outputs.into_iter()) {
                assigned_tensor_map.insert(output_str.clone(), output);
            }
        }

        // Constrain the output
        for (i, cell) in res.iter().enumerate() {
            layouter
                .constrain_instance(cell.as_ref().cell(), config.public, i)
                .unwrap();
        }
        Ok(())
    }
}
