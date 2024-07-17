use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance},
};
use ndarray::{s, Array, IxDyn};

use crate::{
    numerics::{
        accumulator::AccumulatorChip,
        dot::DotChip,
        numeric::{Numeric, NumericConfig},
    },
    utils::helpers::{AssignedTensor, CellRc, FieldTensor, NUMERIC_CONFIG},
};

#[derive(Clone, Debug)]
pub struct FullyConnectedConfig<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

pub struct FullyConnectedCircuit<F: PrimeField> {
    pub input: FieldTensor<F>,
    pub weight: FieldTensor<F>,
}

impl<F: PrimeField> FullyConnectedCircuit<F> {
    pub fn construct(input: FieldTensor<F>, weight: FieldTensor<F>) -> Self {
        Self { input, weight }
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
                match Array::from_shape_vec(
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
                        .collect::<Result<Vec<_>, Error>>()?,
                ) {
                    Ok(x) => Ok(x),
                    Err(e) => panic!(
                        "Error occurs at FullyConnectedCircuit.assign_tensor: {:?}",
                        e
                    ),
                }
            },
        )?)
    }

    pub fn assign_constant(
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
}

impl<F: PrimeField> Circuit<F> for FullyConnectedCircuit<F> {
    type Config = FullyConnectedConfig<F>;
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
        let numeric_config = NumericConfig {
            columns,
            constants,
            ..numeric_config
        };

        // Configure numeric chips
        let numeric_config = AccumulatorChip::<F>::configure(meta, numeric_config);
        let numeric_config = DotChip::<F>::configure(meta, numeric_config);

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
        // Check input and weight shapes
        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();
        assert_eq!(input_shape.len(), 2);
        assert_eq!(weight_shape.len(), 2);
        assert_eq!(input_shape[1], weight_shape[0]);

        // Construct dot chip
        let config_rc = config.numeric_config.clone();
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
                let output = dot_chip
                    .compute(
                        layouter.namespace(|| format!("dot_{}_{}", i, j)),
                        &vec![input, weight],
                        &vec![constants.get(&0).unwrap()],
                    )
                    .unwrap();
                outputs.extend(output.into_iter());
            }
        }

        // Constrain public output
        let mut public_layouter = layouter.namespace(|| "public");
        for (i, cell) in outputs.iter().enumerate() {
            public_layouter.constrain_instance(cell.cell(), config.public, i)?;
        }

        Ok(())
    }
}

#[test]
fn test_fully_connected() {
    use crate::utils::helpers::to_field;
    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

    type F = Fr;

    // original vector
    let v_input: Vec<i64> = vec![1; 10];
    let v_hidden_layer: Vec<i64> = vec![1; 100];

    // original matrix
    let o_input = Array::<i64, ndarray::Dim<_>>::from_shape_vec([1, 10], v_input.clone()).unwrap();
    let o_hidden_layer =
        Array::<i64, ndarray::Dim<_>>::from_shape_vec([10, 10], v_hidden_layer.clone()).unwrap();
    let o_output = o_input.dot(&o_hidden_layer);
    println!("{:?}", o_input);
    println!("{:?}", o_hidden_layer);
    println!("{:?}", o_output);

    // field matrix
    let f_input = Array::from_shape_vec(
        IxDyn(o_input.shape()),
        v_input
            .iter()
            .map(|x| to_field::<F>(*x))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let f_hidden_layer = Array::from_shape_vec(
        IxDyn(o_hidden_layer.shape()),
        v_hidden_layer
            .iter()
            .map(|x| to_field::<F>(*x))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let f_output = Array::from_shape_vec(
        IxDyn(o_output.shape()),
        o_output
            .iter()
            .map(|x| to_field::<F>(*x))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    // let input = f_input.clone().into_iter().collect::<Vec<_>>();
    let output = f_output.clone().into_iter().collect::<Vec<_>>();

    let circuit = FullyConnectedCircuit::construct(f_input, f_hidden_layer);

    let k = 10;
    let scale_factor = 1 << 9;

    let nconfig = &NUMERIC_CONFIG;
    let cloned = nconfig.lock().unwrap().clone();
    *nconfig.lock().unwrap() = NumericConfig {
        k,
        scale_factor,
        num_rows: (1 << k) - 10 + 1,
        num_cols: 10,
        use_selectors: true,
        ..cloned
    };

    let prover = MockProver::run(10, &circuit, vec![output]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
