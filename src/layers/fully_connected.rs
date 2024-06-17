use std::{
    collections::{BTreeSet, HashMap},
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, ErrorFront},
};
use ndarray::{s, Array, IxDyn, ShapeError};

use crate::numerics::{
    adder::AdderChip,
    dot::DotChip,
    numeric::{Numeric, NumericType, _NumericConfig},
};

use super::layer::{
    AssignedTensor, CellRc, ConfigLayer, FieldTensor, Layer, LayerConfig, NumericConsumer, Tensor,
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
    // pub config: LayerConfig<F>,
    pub input: FieldTensor<F>,
    pub weight: FieldTensor<F>,
}

impl<F: PrimeField> FullyConnectedCircuit<F> {
    pub fn construct(
        // config: LayerConfig<F>,
        input: FieldTensor<F>,
        weight: FieldTensor<F>,
    ) -> Self {
        Self {
            // config,
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
                                Ok(Rc::new(region.assign_advice(
                                    || "assign tensor cell",
                                    columns[col_idx],
                                    row_idx,
                                    || Value::known(*cell),
                                )?))
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
                            Ok(Rc::new(region.assign_advice(
                                || "assign tensor cell",
                                columns[col_idx],
                                row_idx,
                                || Value::known(*cell),
                            )?))
                        })
                        .collect::<Result<Vec<_>, ErrorFront>>()?,
                )
                .unwrap())
            },
        )?)
    }

    pub fn assign_constant(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<_NumericConfig>,
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

                // TODO: I've made some very bad life decisions
                // TOOD: this needs to be a random oracle
                // let r_base = F::from(0x123456789abcdef);
                // let mut r = r_base.clone();
                // for i in 0..self.num_random {
                //     let rand = region.assign_fixed(
                //         || format!("rand_{}", i),
                //         gadget_config.fixed_columns[0],
                //         constants.len(),
                //         || Value::known(r),
                //     )?;
                //     r = r * r_base;
                //     constants.insert(RAND_START_IDX + (i as i64), Rc::new(rand));
                // }

                Ok(constants)
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
        let k = 10;
        let columns = (0..10).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for col in columns.iter() {
            meta.enable_equality(*col);
        }
        let constants = vec![meta.fixed_column()];
        for cst in constants.iter() {
            meta.enable_equality(*cst);
        }
        let public = meta.instance_column();
        meta.enable_equality(public);

        let numeric_config = _NumericConfig {
            used_numerics: Arc::new(BTreeSet::new()),
            k,
            scale_factor: 1,
            num_rows: (1 << k),
            num_cols: 10,
            columns,
            constants,
            public,
            use_selectors: true,
            selectors: HashMap::new(),
        };

        let numeric_config = AdderChip::<F>::configure(meta, numeric_config);

        DotChip::<F>::configure(meta, numeric_config)
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
        let config_rc = Rc::new(config);
        let dot_chip = DotChip::<F>::construct(config_rc.clone());

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

        // Assign constants
        let constants = self
            .assign_constant(layouter.namespace(|| "assign_constants"), config_rc.clone())
            .unwrap();

        println!("input shape: {:?}", input.shape());
        println!("weight shape: {:?}", weight.shape());
        // println!("constants: {:?}", constants.get(&0).unwrap());

        // Forward pass
        let mut outputs = vec![];
        for i in 0..input_shape[0] {
            for j in 0..weight_shape[1] {
                let input = input
                    .slice(s![i, ..])
                    .into_iter()
                    .map(|x| x.as_ref())
                    .collect::<Vec<_>>();
                let weight = weight
                    .slice(s![.., j])
                    .into_iter()
                    .map(|x| x.as_ref())
                    .collect::<Vec<_>>();
                // println!("input len: {} input: {:?}", input.len(), input);
                // println!("-------------------------------------------------------------");
                // println!("weight len: {} weight: {:?}", weight.len(), weight);
                let output = dot_chip
                    .forward(
                        layouter.namespace(|| format!("dot_{}_{}", i, j)),
                        &vec![input, weight],
                        &vec![constants.get(&0).unwrap()],
                    )
                    .unwrap();
                outputs.extend(output.into_iter());
            }
        }

        println!("*****************************************************************");
        println!("outputs len: {} outputs: {:?}", outputs.len(), outputs);
        println!("*****************************************************************");

        // Constrain public output
        let mut public_layouter = layouter.namespace(|| "public");
        for (i, cell) in outputs.iter().enumerate() {
            public_layouter.constrain_instance(cell.cell(), dot_chip.config.public, i)?;
        }

        Ok(())
    }
}
