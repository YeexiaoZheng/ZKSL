use super::assign::Assign;
use crate::operation::add::AddChip;
use crate::operation::concat::ConcatChip;
use crate::operation::conv::ConvChip;
use crate::operation::max_pool::MaxPoolChip;
use crate::operation::reshape::ReshapeChip;
use crate::operation::unsqueeze::UnsqueezeChip;
use crate::{
    graph::Graph,
    numeric::{NumericConfig, NumericType},
    operation::{
        gather::GatherChip, gemm::GemmChip, mean::MeanChip, none::NoneChip, relu::ReLUChip,
        softmax::SoftMaxChip, OPType, Operation,
    },
    utils::{
        helpers::{
            configure_static, get_circuit_numeric_config, get_numeric_config, to_field,
            FieldTensor, Tensor,
        },
        matcher::{
            match_backward, match_configure, match_consumer, match_load_lookups, match_op_type,
        },
    },
};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    halo2curves::ff::{FromUniformBytes, PrimeField},
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};
use log::{debug, info};
use ndarray::{Array, ShapeError};
use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    rc::Rc,
};

#[derive(Clone, Debug, Default)]
pub struct BackwardCircuit<F: PrimeField + Ord + FromUniformBytes<64>> {
    pub graph: Graph,
    pub used_numerics: BTreeSet<NumericType>,
    pub field_tensor_map: BTreeMap<String, FieldTensor<F>>,
}

#[derive(Clone, Debug)]
pub struct BackwardConfig<F: PrimeField + Ord + FromUniformBytes<64>> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField + Ord + FromUniformBytes<64>> BackwardCircuit<F> {
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

    pub fn run(&mut self) -> Result<Tensor, ShapeError> {
        let mut res = self.graph.tensor_map.get("gradient").unwrap().clone();
        debug!("gradient:{}", res);
        let numeric_config = get_numeric_config();

        for node in self.graph.nodes.iter().rev() {
            if match_op_type(node.op_type.clone()) == OPType::FM {
                continue;
            }
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
                        None => panic!("Tensor not found: '{}'", x),
                    })
                    .collect::<Vec<Tensor>>(),
                &numeric_config,
                &node.attributes,
            )?;
            // println!("outputs:{:?}", outputs);

            res = outputs[0].clone();

            for (output_str, output) in node.backward_outputs.iter().zip(outputs.into_iter()) {
                // println!("output_str:{}", output_str);
                // Update the weight
                self.graph.tensor_map.insert(output_str.clone(), output);
            }
        }

        self.configure_numeric();

        Ok(res)
    }

    pub fn configure_numeric(&self) -> NumericConfig {
        configure_static(NumericConfig {
            used_numerics: self.used_numerics.clone().into(),
            ..get_numeric_config().clone()
        })
    }

    pub fn commitment_tuples(&self) -> Vec<(usize, usize)> {
        let input_str = self.graph.nodes[self.graph.nodes.len() - 1].backward_inputs[0].clone();
        let output_str = self.graph.nodes[0].backward_outputs[0].clone();

        let input_start = 0;
        let input = self.graph.tensor_map.get(&input_str).unwrap().len();

        let params_start = input_start + input;
        let params: usize = self
            .field_tensor_map
            .iter()
            .filter(|(k, _)| k.ends_with(".weight") || k.ends_with(".bias"))
            .map(|(_, v)| v.len())
            .sum();

        let others: usize = self
            .field_tensor_map
            .iter()
            .filter(|(k, _)| {
                !k.ends_with(".weight") && !k.ends_with(".bias") && !k.ends_with(&input_str)
            })
            .map(|(_, v)| v.len())
            .sum();

        let output_start = params_start + params + others;
        let output = self.graph.tensor_map.get(&output_str).unwrap().len();

        info!(
            "input: {}, params: {}, others: {}, output: {}",
            input, params, others, output
        );
        vec![
            (input_start, input),
            (params_start, params),
            (output_start, output),
            (output_start + output, params),
        ]
    }
}

impl<F: PrimeField + Ord + FromUniformBytes<64>> Assign<F> for BackwardCircuit<F> {}

impl<F: PrimeField + Ord + FromUniformBytes<64>> Circuit<F> for BackwardCircuit<F> {
    type Config = BackwardConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Default::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // Get numeric config from global state
        let numeric_config = get_circuit_numeric_config(meta);

        // If used_numerics has gather then add gather advice lookup
        let (gather_lookup, gather_selector) =
            if numeric_config.used_numerics.contains(&NumericType::Gather) {
                let gather_lookup = vec![meta.advice_column(), meta.advice_column()];
                for col in gather_lookup.iter() {
                    meta.enable_equality(*col);
                }
                (gather_lookup, Some(meta.complex_selector()))
            } else {
                (vec![], None)
            };

        // Create assigned columns
        let assigned = (0..numeric_config.assigned_num_cols)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        for col in assigned.iter() {
            meta.enable_equality(*col);
        }

        // Create columns & constants
        let columns = (0..numeric_config.num_cols)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        for col in columns.iter() {
            meta.enable_equality(*col);
        }
        let constants = vec![meta.fixed_column()];
        for col in constants.iter() {
            meta.enable_equality(*col);
        }

        // Update numeric config
        let mut numeric_config = NumericConfig {
            assigned,
            columns,
            constants,
            gather_lookup,
            gather_selector,
            ..numeric_config
        };
        // Add NaturalLookUp to the used numerics
        numeric_config
            .used_numerics
            .insert(NumericType::NaturalLookUp);
        // Add Update to the used numerics because it used in the backward pass to update the weights
        numeric_config.used_numerics.insert(NumericType::Update);

        // Configure each numerics
        let iter =
            <BTreeSet<NumericType> as Clone>::clone(&numeric_config.used_numerics).into_iter();
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
    ) -> Result<(), Error> {
        // Assign tensors
        let input = self
            .field_tensor_map
            .clone()
            .into_iter()
            .filter(|(k, _)| k.ends_with("gradient"))
            .collect::<BTreeMap<_, _>>();
        let params = self
            .field_tensor_map
            .clone()
            .into_iter()
            .filter(|(k, _)| k.ends_with(".weight") || k.ends_with(".bias"))
            .collect::<BTreeMap<_, _>>();
        let others = self
            .field_tensor_map
            .clone()
            .into_iter()
            .filter(|(k, _)| {
                !k.ends_with(".weight") && !k.ends_with(".bias") && !k.ends_with("gradient")
            })
            .collect::<BTreeMap<_, _>>();
        let assigned_input = self
            .assign_tensor_map(
                layouter.namespace(|| "assign_tensor_map"),
                &config.numeric_config.assigned,
                &input,
            )
            .unwrap();
        let assigned_params = self
            .assign_tensor_map(
                layouter.namespace(|| "assign_tensor_map"),
                &config.numeric_config.assigned,
                &params,
            )
            .unwrap();
        let assigned_others = self
            .assign_tensor_map(
                layouter.namespace(|| "assign_tensor_map"),
                &config.numeric_config.assigned,
                &others,
            )
            .unwrap();

        let mut assigned_tensor_map = assigned_input
            .into_iter()
            .chain(assigned_params.clone().into_iter())
            .chain(assigned_others.clone().into_iter())
            .collect::<BTreeMap<_, _>>();

        let mut changed_tensor_map = assigned_params.clone();

        // Assign constants
        let constants = self
            .assign_constants(
                layouter.namespace(|| "assign_constants"),
                config.numeric_config.clone(),
            )
            .unwrap();

        // Assign random
        let random = self
            .assign_random(
                layouter.namespace(|| "assign_random"),
                config.numeric_config.clone(),
            )
            .unwrap();

        // If have embedding then use random to initialize embedding lookup
        // let embeddings = self
        //     .field_tensor_map
        //     .clone()
        //     .into_iter()
        //     .filter(|(k, _)| k.starts_with("embedding"))
        //     .collect::<BTreeMap<_, _>>();
        // let embeddings = embeddings.into_iter().map(|(_, v)| v).collect::<Vec<_>>();
        // let embeddings = embeddings
        //     .iter()
        //     .map(|emb| {
        //         let emb_dim = emb.shape()[1];
        //         let random = frandom[0..emb_dim].to_vec();
        //         emb.outer_iter()
        //             .map(|row| self.dot_vector_f(row.iter().map(|x| *x).collect(), random.clone()))
        //             .collect::<Vec<_>>()
        //     })
        //     .collect::<Vec<_>>();

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
        let input_str = self.graph.nodes[self.graph.nodes.len() - 1].backward_inputs[0].clone();
        let mut res = assigned_tensor_map.get(&input_str).unwrap().clone();
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
                // FM layer is not supported in the backward pass
                OPType::FM => continue,
                OPType::Gather => GatherChip::<F>::construct(config.numeric_config.clone())
                    .backward(
                        layouter.namespace(|| op.op_type.clone()),
                        &inputs,
                        &constants,
                        &random,
                        &op.attributes,
                    ),
                OPType::Unsqueeze => UnsqueezeChip::<F>::construct(config.numeric_config.clone())
                    .backward(
                        layouter.namespace(|| op.op_type.clone()),
                        &inputs,
                        &constants,
                        &random,
                        &op.attributes,
                    ),
                OPType::Concat => ConcatChip::<F>::construct(config.numeric_config.clone())
                    .backward(
                        layouter.namespace(|| op.op_type.clone()),
                        &inputs,
                        &constants,
                        &random,
                        &op.attributes,
                    ),
                OPType::Add => AddChip::<F>::construct(config.numeric_config.clone()).backward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &random,
                    &op.attributes,
                ),
                OPType::Mean => MeanChip::<F>::construct(config.numeric_config.clone()).backward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &random,
                    &op.attributes,
                ),
                OPType::GEMM => GemmChip::<F>::construct(config.numeric_config.clone()).backward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &random,
                    &op.attributes,
                ),
                OPType::ReLU => ReLUChip::<F>::construct(config.numeric_config.clone()).backward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &random,
                    &op.attributes,
                ),
                OPType::SoftMax => SoftMaxChip::<F>::construct(config.numeric_config.clone())
                    .backward(
                        layouter.namespace(|| op.op_type.clone()),
                        &inputs,
                        &constants,
                        &random,
                        &op.attributes,
                    ),
                OPType::Conv => ConvChip::<F>::construct(config.numeric_config.clone()).backward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &random,
                    &op.attributes,
                ),
                OPType::MaxPool => MaxPoolChip::<F>::construct(config.numeric_config.clone())
                    .backward(
                        layouter.namespace(|| op.op_type.clone()),
                        &inputs,
                        &constants,
                        &random,
                        &op.attributes,
                    ),
                OPType::Reshape => ReshapeChip::<F>::construct(config.numeric_config.clone())
                    .backward(
                        layouter.namespace(|| op.op_type.clone()),
                        &inputs,
                        &constants,
                        &random,
                        &op.attributes,
                    ),
                OPType::None => NoneChip::<F>::construct(config.numeric_config.clone()).backward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &random,
                    &op.attributes,
                ),
                _ => panic!("Operation type not supported"),
            }
            .unwrap();
            res = outputs[0].clone();
            // Insert the output to the assigned tensor map
            for (output_str, output) in op.backward_outputs.iter().zip(outputs.into_iter()) {
                assigned_tensor_map.insert(output_str.clone(), output.clone());
                // Update the weight
                if output_str.ends_with(".weight") || output_str.ends_with(".bias") {
                    changed_tensor_map.insert(output_str.clone(), output.clone());
                }
            }
            debug!(
                "op: {:?}\t backward circuit compute successfully!",
                op.op_type
            );
        }

        // Output
        let output_str = self.graph.nodes[0].backward_outputs[0].clone();
        let output = assigned_tensor_map.get(&output_str).unwrap().clone();
        debug!("output: {:?}", output);

        // Assign the output
        self.copy_assign_vector(
            layouter.namespace(|| "assign output"),
            &config.numeric_config.assigned,
            &output.iter().map(|x| x.as_ref()).collect::<Vec<_>>(),
        )?;

        // Assign the changed tensor map
        self.copy_assign_tensor_map(
            layouter.namespace(|| "assign changed tensor map"),
            &config.numeric_config.assigned,
            &changed_tensor_map,
        )?;

        // Constrain the output
        for (i, cell) in res.iter().enumerate() {
            layouter
                .constrain_instance(cell.as_ref().cell(), config.public, i)
                .unwrap();
        }

        Ok(())
    }
}
