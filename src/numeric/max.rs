use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};

use crate::utils::helpers::{to_field, to_primitive};

use super::{NumericConfig, NumericLayout, NumericType};

type MaxConfig = NumericConfig;

pub struct MaxLayouter<F: PrimeField> {
    pub config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> MaxLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, numeric_config: NumericConfig) -> MaxConfig {
        let selector = meta.complex_selector();
        let columns = &numeric_config.columns;
        let num_inputs = columns.len() - 1;

        /*
            *      [input1]       |  max_row1
            *      [input2]       |  max_row2
            * max_row1  max_row2  |    max
            *
            *
            * max([1,3,-5,6,-2]) = 6

            * col_len = 4
            *  填充:[1,3,-5,6,-2] => [1,3,-5,6,-2,  1]
            *  1  3  -5   |   3
            *  6  -2  1   |   6
            *
            *  [3,6]  =>  [3,6,  3]
            *  3  6  3  |  6
            *  output = 6
            *
        */

        // Need to constrain (max value - inp1) * (max value - inp2) * ... = 0
        meta.create_gate("max gate", |meta| {
            let s = meta.query_selector(selector);
            let outp = meta.query_advice(columns[num_inputs], Rotation::cur());
            let mut res = s.clone();
            for i in 0..num_inputs {
                let inpt = meta.query_advice(columns[i], Rotation::cur());
                res = res.clone() * (inpt - outp.clone());
            }
            vec![res]
        });

        // Need to judge the max value - inp1, inp2, ... >= 0
        let natural_lookup = numeric_config
            .tables
            .get(&NumericType::NaturalLookUp)
            .unwrap()[0];
        for i in 0..num_inputs {
            meta.lookup("max inpt", |meta| {
                let s = meta.query_selector(selector);
                let outp = meta.query_advice(columns[num_inputs], Rotation::cur());
                let inpt = meta.query_advice(columns[i], Rotation::cur());
                vec![(s.clone() * (outp.clone() - inpt), natural_lookup)]
            });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Max, selector);

        MaxConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for MaxLayouter<F> {
    fn name(&self) -> String {
        "max".to_string()
    }

    fn num_rows_per_unit(&self) -> usize {
        1
    }

    fn num_cols_per_row(&self) -> usize {
        // Check it equals to the number in the configure function
        self.config.columns.len() - 1
    }

    fn layout_unit(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        _constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check input shapes
        assert_eq!(inputs.len(), 1);
        let input = inputs[0].clone();
        if input.len() == 1 {
            return Ok(vec![input[0].clone()]);
        }
        let columns = &self.config.columns;

        // Assign input
        let mut inp: Vec<AssignedCell<F, F>> = vec![];
        input[0].value().map(|f| {
            inp = self
                .assign_row(region, columns, copy_advice, row_offset, &input, Some(*f))
                .unwrap();
        });

        // Enable selectors
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Max).unwrap();
            selector.enable(region, row_offset)?;
        }

        let value = inp
            .iter()
            .map(|cell| cell.value().copied())
            .collect::<Value<Vec<_>>>()
            .and_then(|v| {
                let a = v.into_iter().map(|f| to_primitive(&f)).collect::<Vec<_>>();
                if a.len() == 0 {
                    Value::unknown()
                } else {
                    Value::known(to_field::<F>(a.iter().max().cloned().unwrap()))
                }
            });

        let res = region.assign_advice(|| "", columns[columns.len() - 1], row_offset, || value)?;

        Ok(vec![res])
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
        // Only one input is expected
        assert_eq!(inputs.len(), 1);
        let input = inputs[0].clone();
        let cols_per_row = self.num_cols_per_row();

        // Calculate the number of units
        let used_units = self.used_units(input.len());

        let mut row_offset = row_offset;

        // Get initial outputs
        let mut outputs = input
            .chunks(cols_per_row)
            .enumerate()
            .map(|(i, chunk)| {
                self.layout_unit(
                    region,
                    row_offset + i * rows_per_unit,
                    copy_advice,
                    &vec![chunk.to_vec()],
                    constants,
                )
                .unwrap()
                .first()
                .unwrap()
                .clone()
            })
            .collect::<Vec<_>>();

        row_offset += used_units * rows_per_unit;

        // Compute until there is only one output
        while outputs.len() > 1 {
            let used_units = self.used_units(outputs.len());
            outputs = outputs
                .chunks(cols_per_row)
                .enumerate()
                .map(|(i, chunk)| {
                    self.layout_unit(
                        region,
                        row_offset + i,
                        true,
                        &vec![chunk.iter().collect::<Vec<_>>()],
                        constants,
                    )
                    .unwrap()
                    .first()
                    .unwrap()
                    .clone()
                })
                .collect::<Vec<_>>();
            row_offset += used_units * rows_per_unit;
        }

        Ok((outputs, row_offset))
    }
}

#[cfg(test)]
mod tests {
    use std::{marker::PhantomData, rc::Rc, vec};

    use crate::{
        numeric::{
            max::MaxLayouter, nonlinear::natural::NaturalLookUp, NumericConfig, NumericLayout,
            NumericType,
        },
        stage::assign::Assign,
        utils::{
            helpers::{configure_static, get_circuit_numeric_config, to_field, FieldTensor},
            matcher::match_load_lookups,
            math::Int,
        },
    };
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        dev::MockProver,
        halo2curves::{bn256::Fr, ff::PrimeField},
        plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
    };
    use ndarray::{s, Array, IxDyn};

    #[derive(Clone, Debug)]
    pub struct MaxLayerConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    pub struct MaxLayerCircuit<F: PrimeField> {
        pub input1: FieldTensor<F>,
    }

    impl<F: PrimeField> MaxLayerCircuit<F> {
        pub fn construct(input1: FieldTensor<F>) -> Self {
            Self { input1 }
        }
    }

    impl<F: PrimeField> Assign<F> for MaxLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for MaxLayerCircuit<F> {
        type Config = MaxLayerConfig<F>;
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
            println!("Configure numeric chips......");
            let numeric_config = NaturalLookUp::<F>::configure(meta, numeric_config);
            let numeric_config = MaxLayouter::<F>::configure(meta, numeric_config);

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
            // Construct max layer chip
            let config_rc = config.numeric_config.clone();
            let max_layouter = MaxLayouter::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input1 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &max_layouter.config.columns,
                &self.input1,
            )?;
            let input1 = input1
                .slice(s![..])
                .into_iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>();

            let input = &vec![input1];

            // Load lookups
            match_load_lookups(
                config.numeric_config.clone(),
                NumericType::NaturalLookUp,
                layouter.namespace(|| "load field lookups"),
            )
            .unwrap();

            // Assign constants
            let constants = self
                .assign_constants(layouter.namespace(|| "assign_constants"), config_rc.clone())
                .unwrap();
            let zero = constants.get(&0).unwrap().clone();
            let constants = vec![zero.as_ref()];

            // println!("hhhh");

            // Forward pass
            let mut outputs = vec![];
            outputs.extend(layouter.assign_region(
                || "max layer",
                |mut region| {
                    Ok(max_layouter
                        .layout(&mut region, 0, input, &constants)
                        .unwrap()
                        .0)
                },
            )?);

            println!("outputs: {:?}", outputs);

            // Constrain public output
            let mut public_layouter = layouter.namespace(|| "public");
            for (i, cell) in outputs.iter().enumerate() {
                public_layouter.constrain_instance(cell.cell(), config.public, i)?;
            }

            Ok(())
        }
    }

    #[test]
    fn test_max_circuit() {
        // Parameters
        let scale_factor = 1024;
        let k = 18;
        let num_cols = 4;
        configure_static(NumericConfig {
            k,
            num_cols,
            scale_factor,
            max_val: (1 << (k - 2)) - 1,
            min_val: -(1 << (k - 2)),
            batch_size: 1,
            use_selectors: true,
            ..Default::default()
        });

        type F = Fr;
        // 定义测试输入和输出
        let input1: Vec<Int> = vec![1, 2, 3, 4, 5, 10];
        let expected_output: Vec<Int> = vec![10];

        // let input1:Vec<Int> = vec![1,2,3];
        // let input2:Vec<Int> = vec![2,3,4];
        // let expected_output:Vec<Int> = vec![3,5,7];

        let input1 = Array::from_shape_vec(
            IxDyn(&[input1.len()]),
            input1.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>(),
        )
        .unwrap();

        // println!("input1: {:?}", input1);

        let expected_output = Array::from_shape_vec(
            IxDyn(&[expected_output.len()]),
            expected_output
                .iter()
                .map(|x| to_field::<F>(*x))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let expected_output = expected_output.clone().into_iter().collect::<Vec<_>>();

        // 构造电路实例
        let circuit = MaxLayerCircuit::construct(input1);

        // 运行电路验证
        let prover = MockProver::run(k as u32, &circuit, vec![expected_output]).unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }
}
