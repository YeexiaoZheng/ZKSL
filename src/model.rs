use std::{collections::BTreeSet, marker::PhantomData, sync::Mutex};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::ff::{FromUniformBytes, PrimeField},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, ErrorFront, Instance},
};
use ndarray::{Array, Dim, IxDyn};

use crate::{
    layers::layer::{AssignedTensor, LayerConfig, LayerType},
    numerics::{dot::DotChip, numeric::{NumericConfig, NumericType}},
    utils::matcher::{match_layer_name_to_layer_type, match_layer_type_to_consumer},
};

use lazy_static::lazy_static;
lazy_static! {
    pub static ref NUMERIC_CONFIG: Mutex<NumericConfig> = Mutex::new(NumericConfig::default());
    // pub static ref PUBLIC_VALS: Mutex<Vec<BigUint>> = Mutex::new(vec![]);
}

#[derive(Clone, Debug, Default)]
pub struct FormatLayer<F: PrimeField> {
    pub layername: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub weight_shape: Vec<usize>,
    pub original_weights: Array<i64, Dim<[usize; 2]>>,
    pub field_weights: Array<F, IxDyn>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelCircuit<F: PrimeField> {
    pub k: usize,
    pub layers: Vec<FormatLayer<F>>,
    pub layer_chips: Vec<LayerType>,
    pub layer_configs: Vec<LayerConfig>,
    pub used_numerics: BTreeSet<NumericType>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig<F: PrimeField> {
    pub numeric_config: NumericConfig,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ModelCircuit<F> {
    pub fn construct(k: usize, layers: Vec<FormatLayer<F>>) -> Self {
        let mut layer_configs = vec![];
        let mut layer_chips = vec![];
        let mut used_numerics = BTreeSet::new();
        for layer in layers.iter() {
            let layer_params = vec![];
            let layer_type = match_layer_name_to_layer_type(layer.layername.clone());
            layer_configs.push(LayerConfig {
                layer_type,
                layer_params: layer_params.clone(),
                input_shape: layer.input_shape.clone(),
                output_shape: layer.output_shape.clone(),
                mask: vec![],
            });
            layer_chips.push(layer_type);
            used_numerics.extend(
                match_layer_type_to_consumer::<F>(layer_type)
                    .used_numerics(layer_params)
                    .iter(),
            )
        }
        Self {
            k,
            layers,
            layer_chips,
            layer_configs,
            used_numerics,
        }
    }

    pub fn assign_tensors(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensors: &Vec<Array<F, IxDyn>>,
    ) -> Result<Vec<AssignedTensor<F>>, Error> {
        Ok(layouter.assign_region(
            || "assign_tensors",
            |mut region| {
                let mut cell_idx = 0;
                let assigned_tensors = tensors
                    .iter()
                    .map(|tensor| {
                        let assigned_tensor = tensor
                            .iter()
                            .map(|cell| {
                                let row_idx = cell_idx / columns.len();
                                let col_idx = cell_idx % columns.len();
                                let cell = region.assign_advice(
                                    || "assign tensor cell",
                                    columns[col_idx],
                                    row_idx,
                                    || Value::known(*cell),
                                )?;
                                cell_idx += 1;
                                Ok(cell)
                            })
                            .collect::<Result<Vec<_>, ErrorFront>>()?;
                        Ok(Array::from_shape_vec(IxDyn(tensor.shape()), assigned_tensor).unwrap())
                    })
                    .collect::<Result<Vec<_>, ErrorFront>>()?;
                Ok(assigned_tensors)
            },
        )?)
    }
}

impl<F: PrimeField> Circuit<F> for ModelCircuit<F> {
    type Config = ModelConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        todo!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let mut numeric_config = NUMERIC_CONFIG.lock().unwrap().clone();
        let binding = numeric_config.used_numerics.clone();
        let iter = binding.iter();
        for numeric in iter {
            numeric_config = match numeric {
                NumericType::Dot => DotChip::<F>::configure(meta, numeric_config.clone()),
            };
        }
        todo!()
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        // assign tensors
        self.assign_tensors(
            layouter.namespace(|| "assign_tensors"),
            &config.numeric_config.columns,
            &self
                .layers
                .iter()
                .map(|layer| layer.field_weights.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        Ok(())
    }
}
