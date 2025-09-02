// input lookup
pub mod natural;

pub mod exp;
pub mod gather;
pub mod ln;
pub mod relu;

// non-linear numeric trait are defined here
use std::{collections::BTreeMap, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error, Expression, Selector},
    poly::Rotation,
};

use crate::{
    numeric::{NumericConfig, NumericLayout, NumericType},
    utils::{
        helpers::{to_field, to_primitive},
        math::Int,
    },
};

const NUM_LOOKUP_ROWS_PER_UNIT: usize = 6;

pub trait NonLinearNumericLayout<F: PrimeField>: NumericLayout<F> {
    fn num_rows_per_unit() -> usize {
        NUM_LOOKUP_ROWS_PER_UNIT
    }

    fn generate_map(scale_factor: u64, min_val: Int, max_val: Int) -> BTreeMap<Int, Int>;

    fn get_numeric_config(&self) -> Rc<NumericConfig>;

    fn get_numeric_type(&self) -> NumericType;

    fn get_val_in_map(&self, key: Int) -> Int {
        match self.get_numeric_config().maps.get(&self.get_numeric_type()) {
            Some(map) => match map.get(&key) {
                Some(val) => *val,
                None => panic!("Value '{}' not found in map", key),
            }
            .clone(),
            None => panic!("Map {:?} is not found", self.get_numeric_type()),
        }
    }

    fn get_selector(&self) -> Selector {
        match self
            .get_numeric_config()
            .selectors
            .get(&self.get_numeric_type())
        {
            Some(&selectors) => selectors,
            None => panic!("Selector {:?} is not found", self.get_numeric_type()),
        }
    }

    fn _configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
        numeric_type: NumericType,
        shift: bool,
    ) -> NumericConfig {
        let selector = meta.complex_selector();
        let columns = numeric_config.columns;

        let mut tables = numeric_config.tables;
        let input_lookup = NumericType::NaturalLookUp;
        let input_lookup = match tables.get(&input_lookup) {
            Some(tables) => tables,
            None => panic!("Input {:?} table not found", input_lookup),
        }[0];
        let output_lookup = meta.lookup_table_column();

        for idx in 0..columns.len() / 2 {
            let inp_offset = idx;
            let out_offset = idx + columns.len() / 2;
            // let format = format!("non-linear: {:?} lookup", numeric_type);
            meta.lookup("non-linear", |meta| {
                let s = meta.query_selector(selector);
                let mut input_col = meta.query_advice(columns[inp_offset], Rotation::cur());
                if shift {
                    let shift_f = F::from(-numeric_config.min_val as u64);
                    input_col = input_col + Expression::Constant(shift_f);
                }
                let output_col = meta.query_advice(columns[out_offset], Rotation::cur());
                vec![
                    (s.clone() * input_col, input_lookup),
                    (s.clone() * output_col, output_lookup),
                ]
            });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(numeric_type, selector);

        tables.insert(numeric_type, vec![input_lookup, output_lookup]);

        let mut maps = numeric_config.maps;
        let non_linear_map = Self::generate_map(
            numeric_config.scale_factor,
            numeric_config.min_val,
            numeric_config.max_val,
        );
        maps.insert(numeric_type, non_linear_map);

        NumericConfig {
            columns,
            selectors,
            tables,
            maps,
            ..numeric_config
        }
    }

    fn load_lookups(&self, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let config = self.get_numeric_config();
        let numeric_type = self.get_numeric_type();
        let output_lookup = config.tables.get(&numeric_type).unwrap()[1];

        layouter.assign_table(
            || "non-linear table",
            |mut table| {
                // println!("Loading non-linear table");
                for x in config.min_val..config.max_val {
                    let i = x - config.min_val;
                    let val = to_field::<F>(self.get_val_in_map(x));
                    table.assign_cell(
                        || "non-linear cell",
                        output_lookup,
                        i as usize,
                        || Value::known(val),
                    )?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    fn layout_unit(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        _constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let columns = &self.get_numeric_config().columns;

        // Assign input
        let input = self.assign_row(
            region,
            columns,
            copy_advice,
            row_offset,
            &inputs[0],
            Some(Self::ZERO),
        )?;

        // Enable the selector
        if self.get_numeric_config().use_selectors {
            let selector = self.get_selector();
            selector.enable(region, row_offset).unwrap();
        }

        let res = input
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let value = cell.value().map(|x| {
                    let x = to_primitive::<F>(x);
                    to_field::<F>(self.get_val_in_map(x))
                });
                region.assign_advice(
                    || "non-linear",
                    columns[i + self.num_cols_per_row()],
                    row_offset,
                    || value,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(res)
    }

    fn layout_customise(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        rows_per_unit: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<(Vec<AssignedCell<F, F>>, usize), Error> {
        let input = inputs[0].clone();
        let input_len = input.len();
        let cols_per_row = self.num_cols_per_row();

        // Calculate the number of units needed
        let used_rows = self.used_units(input.len());

        let output = input
            .chunks(cols_per_row)
            .enumerate()
            .map(|(i, chunk)| {
                let row_offset = row_offset + i * rows_per_unit;
                <Self as NumericLayout<F>>::layout_unit(
                    self,
                    region,
                    row_offset,
                    copy_advice,
                    &vec![chunk.to_vec()],
                    constants,
                )
                .unwrap()
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok((
            output[0..input_len].to_vec(),
            row_offset + used_rows * rows_per_unit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::{marker::PhantomData, rc::Rc};

    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        halo2curves::ff::PrimeField,
        plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
    };

    use crate::{
        numeric::{
            nonlinear::{exp::ExpLookUp, ln::LnLookUp, relu::ReluLookUp},
            NumericConfig, NumericLayout, NumericType,
        },
        utils::{helpers::to_field, matcher::match_load_lookups, math::Int},
    };
    use crate::{
        stage::assign::Assign,
        utils::helpers::{configure_static, get_circuit_numeric_config},
    };

    use super::natural::NaturalLookUp;

    #[derive(Clone, Debug)]
    pub struct ExpConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    pub struct ExpLayerCircuit<F: PrimeField> {
        pub input: Vec<F>,
    }

    impl<F: PrimeField> ExpLayerCircuit<F> {
        pub fn construct(input: Vec<F>) -> Self {
            Self { input }
        }
    }

    impl<F: PrimeField> Assign<F> for ExpLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for ExpLayerCircuit<F> {
        type Config = ExpConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            todo!()
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            // Get numeric config from global state
            let numeric_config = get_circuit_numeric_config(meta);

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
            let numeric_config = NaturalLookUp::<F>::configure(meta, numeric_config);
            let numeric_config = ExpLookUp::<F>::configure(meta, numeric_config);
            let numeric_config = ReluLookUp::<F>::configure(meta, numeric_config);
            let numeric_config = LnLookUp::<F>::configure(meta, numeric_config);

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
            let exp_layouter = ExpLookUp::<F>::construct(config_rc.clone());
            let relu_layouter = ReluLookUp::<F>::construct(config_rc.clone());
            // let ln_layouter = LnLookUp::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input = self
                .assign_vector(
                    layouter.namespace(|| "assign_inputs"),
                    &exp_layouter.config.columns,
                    &self.input,
                )
                .unwrap()
                .iter()
                .map(|x| x.as_ref().clone())
                .collect::<Vec<_>>();

            // Load lookups
            match_load_lookups(
                config.numeric_config.clone(),
                NumericType::NaturalLookUp,
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
                .assign_constants(layouter.namespace(|| "assign_constants"), config_rc.clone())
                .unwrap();
            let zero = constants.get(&0).unwrap().clone();
            let one = constants.get(&1).unwrap().clone();
            let constants = vec![zero.as_ref(), one.as_ref()];

            // Forward pass
            let mut outputs = vec![];
            outputs.extend(layouter.assign_region(
                || "Nonlinear layer",
                |mut region| {
                    let (relu_output, row_offset) = relu_layouter
                        .layout(&mut region, 0, &vec![input.iter().collect()], &constants)
                        .unwrap();

                    // println!("relu_output: {:#?}", relu_output);

                    let (exp_output, row_offset) = exp_layouter
                        .layout(
                            &mut region,
                            row_offset,
                            &vec![relu_output.iter().collect()],
                            &constants,
                        )
                        .unwrap();

                    // println!("exp_output: {:#?}", exp_output);

                    let _ = row_offset;

                    Ok(exp_output)

                    // Ok(ln_layouter
                    //     .layout(
                    //         &mut region,
                    //         row_offset,
                    //         &vec![exp_output.iter().collect()],
                    //         &constants,
                    //     )
                    //     .unwrap()
                    //     .0)
                },
            )?);

            // println!("outputs: {:#?}", outputs);

            // Constrain public output
            let mut public_layouter = layouter.namespace(|| "public");
            for (i, cell) in outputs.iter().enumerate() {
                public_layouter.constrain_instance(cell.cell(), config.public, i)?;
            }

            Ok(())
        }
    }

    #[test]
    fn test_nonlinear_circuit() {
        use crate::utils::math::{exp, /*ln,*/ relu};
        use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

        // Parameters
        let k = 15;
        let numeric_config = configure_static(NumericConfig {
            k,
            num_cols: 6,
            scale_factor: 1024,
            use_selectors: true,
            ..Default::default()
        });

        // original vector
        let v_input: Vec<Int> = vec![0, 1, -2, 3, -4, 100];
        let v_relu: Vec<Int> = v_input.iter().map(|x| relu(*x)).collect();
        let v_exp: Vec<Int> = v_relu
            .iter()
            .map(|x| exp(*x, numeric_config.scale_factor))
            .collect();
        // let v_ln: Vec<Int> = v_exp
        //     .iter()
        //     .map(|x| ln(*x, numeric_config.scale_factor))
        //     .collect();
        let v_output: Vec<Int> = v_exp.clone();

        println!("v_input: {:?}", v_input);
        println!("v_relu: {:?}", v_relu);
        println!("v_exp: {:?}", v_exp);
        // println!("v_ln: {:?}", v_ln);
        println!("v_output: {:?}", v_output);

        // field vector
        let f_input = v_input
            .iter()
            .map(|x| to_field::<Fr>(*x))
            .collect::<Vec<_>>();
        let f_output = v_output
            .iter()
            .map(|x| to_field::<Fr>(*x))
            .collect::<Vec<_>>();

        let circuit = ExpLayerCircuit::construct(f_input);

        let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![f_output]).unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }
}
