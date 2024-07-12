use std::{
    collections::{BTreeSet, HashMap},
    marker::PhantomData,
    rc::Rc,
};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::ff::{FromUniformBytes, PrimeField},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance},
};
use ndarray::{Array, IxDyn, ShapeError};

use crate::{
    commitment::poseidon::FixedPoseidonChip,
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
        helpers::{to_field, AssignedTensor, CellRc, FieldTensor, Tensor, NUMERIC_CONFIG},
        matcher::{
            match_configure, match_consumer, match_forward, match_load_lookups, match_op_type,
        },
        math::Int,
    },
    weight::AssignedWeight,
};

#[derive(Clone, Debug)]
pub struct ModelCircuit<F: PrimeField> {
    pub graph: Graph,
    pub used_numerics: BTreeSet<NumericType>,
    pub field_tensor_map: HashMap<String, FieldTensor<F>>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig<F: PrimeField + Ord + FromUniformBytes<64>> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub hasher: Option<FixedPoseidonChip<F>>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ModelCircuit<F> {
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

    pub fn assign_tensor_map(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensors_map: &HashMap<String, FieldTensor<F>>,
    ) -> Result<HashMap<String, AssignedTensor<F>>, Error> {
        Ok(layouter.assign_region(
            || "assign_tensor_map",
            |mut region| {
                let mut cell_idx = 0;
                let assigned_tensors = tensors_map
                    .iter()
                    .map(|(key, tensor)| {
                        let assigned_tensor = tensor
                            .iter()
                            .map(|cell| {
                                let row_idx = cell_idx / columns.len();
                                let col_idx = cell_idx % columns.len();
                                cell_idx += 1;
                                Ok(Rc::new(region.assign_advice(
                                    || "assign tensor cell",
                                    columns[col_idx],
                                    row_idx,
                                    || Value::known(*cell),
                                )?))
                            })
                            .collect::<Result<Vec<_>, Error>>()?;
                        Ok((
                            key.clone(),
                            match Array::from_shape_vec(IxDyn(tensor.shape()), assigned_tensor) {
                                Ok(x) => x,
                                Err(e) => panic!(
                                    "Error occurs at ModelCircuit.assign_tensors_map: {:?}",
                                    e
                                ),
                            },
                        ))
                    })
                    .collect::<Result<HashMap<_, _>, Error>>()?;
                Ok(assigned_tensors)
            },
        )?)
    }

    pub fn assign_constants(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<NumericConfig>,
    ) -> Result<HashMap<Int, CellRc<F>>, Error> {
        let sf = config.scale_factor;
        // let min_val = config.min_val;
        // let max_val = config.max_val;

        Ok(layouter.assign_region(
            || "constants",
            |mut region| {
                let mut constants: HashMap<Int, CellRc<F>> = HashMap::new();

                let vals = vec![0 as Int, 1, sf as Int /*min_val, max_val*/];
                for (i, val) in vals.iter().enumerate() {
                    let cell = region.assign_fixed(
                        || format!("constant_{}", i),
                        config.constants[0],
                        i,
                        || Value::known(to_field::<F>(*val)),
                    )?;
                    constants.insert(*val, Rc::new(cell));
                }

                Ok(constants)
            },
        )?)
    }

    pub fn forward(&self) -> Result<Tensor, ShapeError> {
        let mut tensor_map = self.graph.tensor_map.clone();
        let numeric_config = NUMERIC_CONFIG.lock().unwrap().clone();

        for node in self.graph.nodes.iter() {
            let operation = match_forward::<F>(match_op_type(node.op_type.clone()));
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

impl<F: PrimeField + Ord + FromUniformBytes<64>> Circuit<F> for ModelCircuit<F> {
    type Config = ModelConfig<F>;
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

        ModelConfig {
            numeric_config: Rc::new(numeric_config.clone()),
            public,
            hasher: FixedPoseidonChip::configure(meta, numeric_config.clone()),
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
                    "Error occurs at ModelCircuit.synthesize load lookups: {:?}",
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
        }

        // Constrain the output
        let output = assigned_tensor_map.get("output").unwrap().clone();
        for (i, cell) in output.iter().enumerate() {
            layouter
                .constrain_instance(cell.as_ref().cell(), config.public, i)
                .unwrap();
        }

        // Constrain the weight hash
        let mut hash_outputs = vec![];
        if config.hasher.is_some() {
            let hasher = config.hasher.unwrap();
            let weight = AssignedWeight::<F>::construct(
                self.graph.nodes.clone(),
                assigned_tensor_map.clone(),
            );
            hash_outputs = hasher
                .hash_vec(
                    layouter.namespace(|| "poseidon hasher"),
                    weight.to_vec(),
                    constants[&0].clone(),
                )
                .unwrap();
        }
        // println!("hash_outputs len: {:#?}", hash_outputs.len());
        // println!("hash_outputs: {:?}", hash_outputs);

        let offset = output.len();
        for (i, cell) in hash_outputs.iter().enumerate() {
            layouter
                .constrain_instance(cell.cell(), config.public, i + offset)
                .unwrap();
        }
        Ok(())
    }
}
