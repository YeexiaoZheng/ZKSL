use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error, Expression},
    poly::Rotation,
};

use super::{NumericConfig, NumericLayout, NumericType};

type SumConfig = NumericConfig;

pub struct SumLayouter<F: PrimeField> {
    pub config: Rc<SumConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> SumLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, numeric_config: NumericConfig) -> SumConfig {
        let selector = meta.complex_selector();
        let columns = numeric_config.columns;

        /* sum([1, 2, 3, 4, 5]) = 15
            advice      |  selector
            -----------------------
            inp...  out |   1

            e.g.
            len_col = 6
            1  2   3   4   5   15  |  1
        */

        // Create a add gate for the sum (sum will use it recursively to add all inputs)
        meta.create_gate("sum add gate", |meta| {
            let s = meta.query_selector(selector);

            let gate_input = columns[0..columns.len() - 1]
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::cur()))
                .collect::<Vec<_>>();
            let gate_output = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

            let res = gate_input
                .iter()
                .fold(Expression::Constant(F::ZERO), |acc, x| acc + x.clone());

            vec![s * (res - gate_output)]
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Sum, selector);

        SumConfig {
            columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for SumLayouter<F> {
    fn name(&self) -> String {
        "Sum".to_string()
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
        // Check input shape
        assert_eq!(inputs.len(), 1);
        let columns = &self.config.columns;

        // Assign the input by copy advice
        let input = self.assign_row(
            region,
            columns,
            copy_advice,
            row_offset,
            &inputs[0],
            Some(Self::ZERO),
        )?;

        // Enable the selector
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Sum).unwrap();
            selector.enable(region, row_offset)?;
        }

        // Accumulate the input
        let value = input
            .iter()
            .fold(Value::known(F::ZERO), |acc, x| acc + x.value().copied());

        // Assign the output column
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
        // Only need one input to accumulate
        assert_eq!(inputs.len(), 1);

        let cols_per_row = self.num_cols_per_row();
        let input = inputs[0].clone();

        // Calculate the number of units
        let used_units = self.used_units(input.len());

        let mut row_offset = row_offset;

        // Accumulate the input to get initial outputs
        let mut outputs = input
            .chunks(cols_per_row)
            .enumerate()
            .map(|(i, chunk)| {
                self.layout_unit(
                    region,
                    row_offset + i,
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

        // Accumulate recursively until there is only one output
        while outputs.len() > 1 {
            let used_units = self.used_units(outputs.len());
            outputs = outputs
                .chunks(cols_per_row)
                .enumerate()
                .map(|(i, chunk)| {
                    self.layout_unit(
                        region,
                        row_offset + i,
                        copy_advice,
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
            sum::SumLayouter,
            {NumericConfig, NumericLayout},
        },
        stage::assign::Assign,
        utils::{
            helpers::{configure_static, to_field, FieldTensor, NUMERIC_CONFIG},
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
    pub struct SumLayerConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    pub struct SumLayerCircuit<F: PrimeField> {
        pub input1: FieldTensor<F>,
    }

    impl<F: PrimeField> SumLayerCircuit<F> {
        pub fn construct(input1: FieldTensor<F>) -> Self {
            Self { input1 }
        }
    }

    impl<F: PrimeField> Assign<F> for SumLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for SumLayerCircuit<F> {
        type Config = SumLayerConfig<F>;
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
            let numeric_config = SumLayouter::<F>::configure(meta, numeric_config);

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
            // Construct sum layer chip
            let config_rc = config.numeric_config.clone();
            let sum_layouter = SumLayouter::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input1 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &sum_layouter.config.columns,
                &self.input1,
            )?;
            let input1 = input1
                .slice(s![..])
                .into_iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>();

            let input = &vec![input1];

            // Assign constants
            let constants = self
                .assign_constants(layouter.namespace(|| "assign_constants"), config_rc.clone())
                .unwrap();
            let zero = constants.get(&0).unwrap().clone();
            let constants = vec![zero.as_ref()];

            println!("hhhh");

            // Forward pass
            let mut outputs = vec![];
            outputs.extend(layouter.assign_region(
                || "sum layer",
                |mut region| {
                    Ok(sum_layouter
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
    fn test_sum_circuit() {
        // Parameters
        let scale_factor = 1024;
        let k = 18;
        let num_cols = 6;

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
        let input1: Vec<Int> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let expected_output: Vec<Int> = vec![36];

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
        let circuit = SumLayerCircuit::construct(input1);

        // 运行电路验证
        let prover = MockProver::run(k as u32, &circuit, vec![expected_output]).unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }
}
