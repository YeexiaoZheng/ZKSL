use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance},
};
use ndarray::{Array, IxDyn, ShapeError};

use crate::{
    loss::{loss::Loss, softmax::SoftMaxLossChip},
    numerics::{
        accumulator::AccumulatorChip,
        div::DivChip,
        lookups::{field_lookup::FieldLookUpChip, row_lookup::RowLookUpChip},
        max::MaxChip,
        mul::MulChip,
        nonlinear::{exp::ExpChip, relu::ReluChip},
        numeric::{NumericConfig, NumericType},
        sub::SubChip,
    },
    utils::{
        helpers::{to_field, AssignedTensor, CellRc, FieldTensor, Tensor, NUMERIC_CONFIG},
        matcher::match_load_lookups,
        math::Int,
    },
};

#[derive(Clone, Debug)]
pub struct SoftMaxLossConfig<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub public: Column<Instance>,
    pub _marker: PhantomData<F>,
}

pub struct SoftMaxLossCircuit<F: PrimeField> {
    pub input: FieldTensor<F>,
    pub label: Vec<F>,
}

impl<F: PrimeField> SoftMaxLossCircuit<F> {
    pub fn construct(input: FieldTensor<F>, label: Vec<F>) -> Self {
        Self { input, label }
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

    pub fn assign_vec(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        input: &Vec<F>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        Ok(layouter.assign_region(
            || "assign input",
            |mut region| {
                let mut cell_idx = 0;
                input
                    .iter()
                    .map(|cell| {
                        let row_idx = cell_idx / columns.len();
                        let col_idx = cell_idx % columns.len();
                        cell_idx += 1;
                        let out = region.assign_advice(
                            || "assign tensor cell",
                            columns[col_idx],
                            row_idx,
                            || Value::known(*cell),
                        )?;
                        Ok(out)
                    })
                    .collect::<Result<Vec<_>, Error>>()
            },
        )?)
    }

    pub fn assign_constant(
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

    pub fn compute(
        &self,
        input: &Tensor,
        label: &Vec<Int>,
        numeric_config: &NumericConfig,
    ) -> Result<(Int, Tensor), ShapeError> {
        SoftMaxLossChip::<F>::compute(input, label, numeric_config)
    }
}

impl<F: PrimeField> Circuit<F> for SoftMaxLossCircuit<F> {
    type Config = SoftMaxLossConfig<F>;
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
        let numeric_config = FieldLookUpChip::<F>::configure(meta, numeric_config);
        let numeric_config = RowLookUpChip::<F>::configure(meta, numeric_config);
        let numeric_config = ExpChip::<F>::configure(meta, numeric_config);
        let numeric_config = ReluChip::<F>::configure(meta, numeric_config);
        let numeric_config = MaxChip::<F>::configure(meta, numeric_config);
        let numeric_config = SubChip::<F>::configure(meta, numeric_config);
        let numeric_config = MulChip::<F>::configure(meta, numeric_config);
        let numeric_config = DivChip::<F>::configure(meta, numeric_config);
        let numeric_config = AccumulatorChip::<F>::configure(meta, numeric_config);

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
        // Construct Exp chip
        let config_rc = config.numeric_config.clone();
        let softmax_loss_chip = SoftMaxLossChip::<F>::construct(config_rc.clone());

        // Assign input tensors
        let input = self
            .assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &softmax_loss_chip.numeric_config.columns,
                &self.input,
            )
            .unwrap();
        let label = self
            .assign_vec(
                layouter.namespace(|| "assign_inputs"),
                &softmax_loss_chip.numeric_config.columns,
                &self.label,
            )
            .unwrap();

        // Load lookups
        match_load_lookups(
            config.numeric_config.clone(),
            NumericType::FieldLookUp,
            layouter.namespace(|| "load field lookups"),
        )
        .unwrap();
        match_load_lookups(
            config.numeric_config.clone(),
            NumericType::RowLookUp,
            layouter.namespace(|| "load field lookups"),
        )
        .unwrap();
        match_load_lookups(
            config.numeric_config.clone(),
            NumericType::Relu,
            layouter.namespace(|| "load relu lookups"),
        )
        .unwrap();
        match_load_lookups(
            config.numeric_config.clone(),
            NumericType::Exp,
            layouter.namespace(|| "load exp lookups"),
        )
        .unwrap();

        // Assign constants
        let constants = self
            .assign_constant(layouter.namespace(|| "assign_constants"), config_rc.clone())
            .unwrap();

        // Forward pass
        let outputs = softmax_loss_chip
            .compute(
                layouter.namespace(|| "soft max loss compute"),
                &input.view(),
                &label.into_iter().map(|x| Rc::new(x)).collect(),
                &constants,
            )
            .unwrap();
        // println!("outputs: {:#?}", outputs);
        // println!("public: {:#?}", config.public);

        // Constrain public output
        let mut public_layouter = layouter.namespace(|| "public");
        for (i, cell) in outputs.iter().enumerate() {
            public_layouter.constrain_instance(cell.cell(), config.public, i)?;
        }

        Ok(())
    }
}

#[test]
fn test_softmax_loss_circuit() {
    use crate::utils::helpers::configure_static_numeric_config_default;
    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

    type F = Fr;

    let input = vec![300, 700];
    let input = Array::from_shape_vec([1, 2], input).unwrap().into_dyn();
    let label = vec![0];

    let f_input = input.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_label = label.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>();
    let f_input = Array::from_shape_vec([1, 2], f_input).unwrap().into_dyn();

    let numeric_config = configure_static_numeric_config_default();

    let circuit = SoftMaxLossCircuit::construct(f_input, f_label);

    let (loss, gradient) = circuit.compute(&input, &label, &numeric_config).unwrap();

    println!("loss: {:?}", loss);
    println!("gradient: {:?}", gradient);

    let f_gradient = gradient
        .iter()
        .map(|x| to_field::<F>(*x))
        .collect::<Vec<_>>();

    let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_gradient]).unwrap();

    assert_eq!(prover.verify(), Ok(()));
}
