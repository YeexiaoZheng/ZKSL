use std::{
    collections::{BTreeSet, HashMap},
    hash::Hash,
    marker::PhantomData,
    rc::Rc,
    sync::Mutex,
};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::ff::{FromUniformBytes, PrimeField},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, ErrorFront, Instance},
    poly::ipa::strategy::Accumulator,
};
use ndarray::{Array, Dim, IxDyn};

use crate::{
    graph::{Graph, Node},
    layers::{
        self,
        fully_connected::{FullyConnectedChip, FullyConnectedLayer},
        layer::{self, ConfigLayer, Layer, LayerConfig, LayerType}, // layer::{AssignedTensor, ConfigLayer, FieldTensor, Layer, LayerConfig, LayerType},
    },
    numerics::{
        accumulator::AccumulatorChip,
        dot::DotChip,
        numeric::{NumericConfig, NumericType},
    },
    utils::{
        helpers::{to_field, AssignedTensor, CellRc, FieldTensor, Tensor, NUMERIC_CONFIG},
        matcher::{match_layer_name_to_layer_type, match_layer_type_to_consumer},
    },
};

#[derive(Clone, Debug, Default)]
pub struct FormatLayer<F: PrimeField> {
    pub layer_name: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub weight_shape: Vec<usize>,
    pub original_weights: Array<i64, Dim<[usize; 2]>>,
    pub field_weights: Array<F, IxDyn>,
}

#[derive(Clone, Debug)]
pub struct ModelCircuit<F: PrimeField> {
    pub graph: Graph,
    pub used_numerics: BTreeSet<NumericType>,
    pub field_tensor_map: HashMap<String, FieldTensor<F>>,
    pub layer_chips: Vec<LayerType>,
    pub layer_configs: Vec<LayerConfig<F>>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ModelCircuit<F> {
    pub fn construct(graph: Graph) -> Self {
        let layers = &graph.nodes;
        let mut layer_configs = vec![];
        let mut layer_chips = vec![];
        let mut used_numerics = BTreeSet::new();
        for layer in layers.iter() {
            let layer_type = match_layer_name_to_layer_type(layer.op_type.clone());
            // let layer_config = LayerConfig::<F>::construct(layer.clone());
            // layer_configs.push(layer_config.clone());
            layer_chips.push(layer_type);
            used_numerics.extend(
                match_layer_type_to_consumer::<F>(layer_type)
                    .used_numerics()
                    .iter(),
            )
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
            layer_chips,
            layer_configs,
        }
    }

    pub fn assign_tensor_map(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensors_map: &HashMap<String, FieldTensor<F>>,
    ) -> Result<HashMap<String, AssignedTensor<F>>, Error> {
        Ok(layouter.assign_region(
            || "assign_tensors_map",
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
                            .collect::<Result<Vec<_>, ErrorFront>>()?;
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
                    .collect::<Result<HashMap<_, _>, ErrorFront>>()?;
                Ok(assigned_tensors)
            },
        )?)
    }

    pub fn assign_constants(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<NumericConfig>,
    ) -> Result<HashMap<i64, CellRc<F>>, Error> {
        let sf = config.scale_factor;
        // let min_val = config.min_val;
        let min_val = -(1 << (config.k - 1));
        // let max_val = config.max_val;

        Ok(layouter.assign_region(
            || "constants",
            |mut region| {
                let mut constants: HashMap<i64, CellRc<F>> = HashMap::new();

                let vals = vec![0 as i64, 1, sf as i64 /*min_val, max_val*/];
                let shift_val_i64 = -min_val * 2; // FIXME
                let shift_val_f = F::from(shift_val_i64 as u64);
                for (i, val) in vals.iter().enumerate() {
                    let cell = region.assign_fixed(
                        || format!("constant_{}", i),
                        config.constants[0],
                        i,
                        || Value::known(F::from((val + shift_val_i64) as u64) - shift_val_f),
                    )?;
                    constants.insert(*val, Rc::new(cell));
                }

                Ok(constants)
            },
        )?)
    }

    pub fn forward(&self) -> Tensor {
        let mut tensor_map = self.graph.tensor_map.clone();

        for node in self.graph.nodes.iter() {
            match node.op_type.as_str() {
                "Gemm" => {
                    let layer: FullyConnectedLayer<F> = layers::fully_connected::FullyConnectedLayer::construct(
                        LayerConfig::default(),
                    );
                    let output = layer.forward(
                        node.inputs
                            .iter()
                            .map(|x| tensor_map.get(x).unwrap().clone())
                            .collect::<Vec<Tensor>>(),
                    ).unwrap();
                    for op in node.outputs.iter() {
                        tensor_map.insert(op.clone(), output.clone());
                    }
                }
                _ => panic!("Layer type not supported"),
            }
        }

        tensor_map.get("output").unwrap().clone()
    }
}

impl<F: PrimeField> Circuit<F> for ModelCircuit<F> {
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

        // numeric config
        let binding = numeric_config.used_numerics.clone();
        let iter = binding.iter();
        for numeric in iter {
            numeric_config = match numeric {
                NumericType::Dot => DotChip::<F>::configure(meta, numeric_config),
                NumericType::Accumulator => AccumulatorChip::<F>::configure(meta, numeric_config),
            };
        }

        // Create public column
        let public = meta.instance_column();
        meta.enable_equality(public);

        ModelConfig {
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
        let mut assigned_tensor_map = self
            .assign_tensor_map(
                layouter.namespace(|| "assign_tensor_map"),
                &config.numeric_config.columns,
                &self.field_tensor_map,
            )
            .unwrap();

        let constants = self
            .assign_constants(
                layouter.namespace(|| "assign_constants"),
                config.numeric_config.clone(),
            )
            .unwrap();

        for op in self.graph.nodes.iter() {
            let layer_type = match_layer_name_to_layer_type(op.op_type.clone());
            let layer = match layer_type {
                LayerType::FullyConnected => FullyConnectedChip {
                    config: LayerConfig::default(),
                    numeric_config: config.numeric_config.clone(),
                    _marker: PhantomData,
                },
                _ => panic!("Layer type not supported"),
            };
            //
            let inputs = op
                .inputs
                .iter()
                .map(|x| assigned_tensor_map.get(x).unwrap().view())
                .collect();

            let output = layer
                .forward(
                    layouter.namespace(|| op.op_type.clone()),
                    &inputs,
                    &constants,
                    &op.attributes,
                )
                .unwrap();
            for (op, output) in op.outputs.iter().zip(output.into_iter()) {
                assigned_tensor_map.insert(op.clone(), output);
            }
        }
        Ok(())
    }
}
