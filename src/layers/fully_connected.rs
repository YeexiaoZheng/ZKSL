use std::{
    collections::{BTreeSet, HashMap},
    marker::PhantomData,
    sync::Arc,
};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, ErrorFront},
};
use ndarray::{s, Array, IxDyn, ShapeError};

use crate::numerics::{
    dot::DotChip,
    numeric::{Numeric, NumericType, _NumericConfig},
};

use super::layer::{
    AssignedTensor, ConfigLayer, FieldTensor, Layer, LayerConfig, NumericConsumer, Tensor,
};

#[derive(Clone, Debug, Default)]
pub struct FullyConnectedLayer<F: PrimeField> {
    pub config: LayerConfig<F>,
}

impl<F: PrimeField> FullyConnectedLayer<F> {
    pub fn construct(config: LayerConfig<F>) -> Self {
        Self { config }
    }
}

impl<F: PrimeField> ConfigLayer<F> for FullyConnectedLayer<F> {
    fn config(&self) -> &LayerConfig<F> {
        &self.config
    }

    fn forward(&self, input: Tensor) -> Result<Tensor, ShapeError> {
        assert_eq!(input.ndim(), 2);
        assert_eq!(input.ndim(), self.config.input_shape.len());
        assert_eq!(self.config.input_shape[1], self.config.weight_shape[0]);
        assert_eq!(self.config.weight_shape[1], self.config.output_shape[0]);

        let input_shape = (self.config.input_shape[0], self.config.input_shape[1]);
        let input = input.into_shape(input_shape)?;
        let weight_shape = (self.config.weight_shape[0], self.config.weight_shape[1]);
        let weight = self.config.o_weight.clone().into_shape(weight_shape)?;

        Ok(input.dot(&weight).into_dyn())
    }
}

#[derive(Clone, Debug, Default)]
pub struct FullyConnectedChip<F: PrimeField> {
    pub config: LayerConfig<F>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> Layer<F> for FullyConnectedChip<F> {
    fn _forward(&self, input: Tensor) -> Result<Tensor, ShapeError> {
        Ok(FullyConnectedLayer::construct(self.config.clone()).forward(input)?)
    }

    fn forward(&self) {
        todo!()
    }
}

impl<F: PrimeField> NumericConsumer for FullyConnectedChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        let mut numerics = vec![];
        numerics.push(NumericType::Dot);
        numerics
    }
}

pub struct FullyConnectedCircuit<F: PrimeField> {
    pub config: LayerConfig<F>,
    pub input: FieldTensor<F>,
    pub weight: FieldTensor<F>,
}

impl<F: PrimeField> FullyConnectedCircuit<F> {
    pub fn construct(
        config: LayerConfig<F>,
        input: FieldTensor<F>,
        weight: FieldTensor<F>,
    ) -> Self {
        Self {
            config,
            input,
            weight,
        }
    }

    pub fn assign_tensors(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensors: &Vec<FieldTensor<F>>,
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
                                cell_idx += 1;
                                Ok(region.assign_advice(
                                    || "assign tensor cell",
                                    columns[col_idx],
                                    row_idx,
                                    || Value::known(*cell),
                                )?)
                            })
                            .collect::<Result<Vec<_>, ErrorFront>>()?;
                        Ok(Array::from_shape_vec(IxDyn(tensor.shape()), assigned_tensor).unwrap())
                    })
                    .collect::<Result<Vec<_>, ErrorFront>>()?;
                Ok(assigned_tensors)
            },
        )?)
    }

    pub fn assign_tensor(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensor: &FieldTensor<F>,
    ) -> Result<AssignedTensor<F>, Error> {
        Ok(layouter.assign_region(
            || "assign_tensors",
            |mut region| {
                let mut cell_idx = 0;
                Ok(Array::from_shape_vec(
                    IxDyn(tensor.shape()),
                    tensor
                        .iter()
                        .map(|cell| {
                            let row_idx = cell_idx / columns.len();
                            let col_idx = cell_idx % columns.len();
                            cell_idx += 1;
                            Ok(region.assign_advice(
                                || "assign tensor cell",
                                columns[col_idx],
                                row_idx,
                                || Value::known(*cell),
                            )?)
                        })
                        .collect::<Result<Vec<_>, ErrorFront>>()?,
                )
                .unwrap())
            },
        )?)
    }
}

impl<F: PrimeField> Circuit<F> for FullyConnectedCircuit<F> {
    type Config = _NumericConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        todo!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let k = 3;
        let columns = vec![meta.advice_column()];
        for col in columns.iter() {
            meta.enable_equality(*col);
        }
        let fixed = vec![meta.fixed_column()];
        for fix in fixed.iter() {
            meta.enable_equality(*fix);
        }
        let public = meta.instance_column();
        meta.enable_equality(public);

        DotChip::<F>::configure(
            meta,
            _NumericConfig {
                used_numerics: Arc::new(BTreeSet::new()),
                k,
                scale_factor: 1,
                num_rows: (1 << k),
                num_cols: 10,
                columns,
                fixed,
                public,
                use_selectors: true,
                selectors: HashMap::new(),
            },
        )
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        // Check input and weight shapes
        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();
        assert_eq!(input_shape.len(), 2);
        assert_eq!(weight_shape.len(), 2);
        assert_eq!(input_shape[1], weight_shape[0]);

        // Construct dot chip
        let dot_chip = DotChip::<F>::construct(config);

        // Assign input and weight tensors
        let input = self
            .assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &dot_chip.config.columns,
                &self.input,
            )
            .unwrap();
        let weight = self
            .assign_tensor(
                layouter.namespace(|| "assign_weights"),
                &dot_chip.config.columns,
                &self.weight,
            )
            .unwrap();

        // Forward pass
        let mut outputs = vec![];
        for i in 0..input_shape[0] {
            for j in 0..weight_shape[1] {
                let input = input.slice(s![i, ..]).into_dyn();
                let weight = weight.slice(s![.., j]).into_dyn();
                let output = dot_chip.forward(&vec![input, weight]).unwrap();
                outputs.extend(output.into_iter());
            }
        }

        // Constrain public output
        let mut public_layouter = layouter.namespace(|| "public");
        let mut i = 0;
        for tensor in outputs.iter() {
            for cell in tensor.iter() {
                public_layouter
                    .constrain_instance(cell.cell(), dot_chip.config.public, i)
                    .unwrap();
                i += 1;
            }
        }

        Ok(())
    }
}
