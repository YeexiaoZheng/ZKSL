use std::{marker::PhantomData, rc::Rc};

use super::{NumericConfig, NumericLayout, NumericType};
use halo2_proofs::{
    circuit::{AssignedCell, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};

type SquareConfig = NumericConfig;

pub struct SquareLayouter<F: PrimeField> {
    pub config: Rc<SquareConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> SquareLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> SquareConfig {
        let selector = meta.selector();
        let columns = &numeric_config.columns;
        /* square([1, 2, 3, 4]) = [1, 4, 9, 16]
            advice      |  selector
            -----------------------
            inp         |   1
            out         |   0

           e.g.
           len_col = 5
           1  2   3   4   00      ｜  1
           1  4   9  16   00      ｜  0

        */
        meta.create_gate("square gate", |meta| {
            let s = meta.query_selector(selector);
            (0..columns.len() - 1)
                .into_iter()
                .map(|i| {
                    let inp = meta.query_advice(columns[i], Rotation::cur());
                    let outp = meta.query_advice(columns[i], Rotation::next());
                    s.clone() * (inp.square() - outp)
                })
                .collect::<Vec<_>>()
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Square, selector);

        SquareConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for SquareLayouter<F> {
    fn name(&self) -> String {
        "Square".to_string()
    }

    fn num_rows_per_unit(&self) -> usize {
        2
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
        let columns = &self.config.columns;
        // println!("layout_unit input {:?}", &inputs[0]);
        // Assign input
        let inp = self.assign_row(
            region,
            columns,
            copy_advice,
            row_offset,
            &inputs[0],
            Some(Self::ZERO),
        )?;

        // println!("layout_unit input {:?}", inp);

        // Enable the selector
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Square).unwrap();
            selector.enable(region, row_offset).unwrap();
        }

        let outp = inp
            .iter()
            .enumerate()
            .map(|(idx, cell)| {
                // Calculate value of square
                let value = cell.value().copied() * cell.value().copied();
                region
                    .assign_advice(|| "", columns[idx], row_offset + 1, || value)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        Ok(outp)
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
        // Check shapes
        let input = inputs[0].clone();
        let cols_per_row = self.num_cols_per_row();

        let used_units = self.used_units(input.len());

        let mut outputs = vec![];
        let _ = input
            .chunks(cols_per_row)
            .enumerate()
            .map(|(i, inp)| {
                outputs.extend(
                    self.layout_unit(
                        region,
                        row_offset + i * rows_per_unit,
                        copy_advice,
                        &vec![inp.to_vec()],
                        constants,
                    )
                    .unwrap(),
                );
            })
            .collect::<Vec<_>>();
        let row_offset = row_offset + used_units * rows_per_unit;

        Ok((outputs[0..input.len()].to_vec(), row_offset))
    }
}

#[cfg(test)]
mod tests {
    use std::{marker::PhantomData, rc::Rc, vec};

    use crate::{
        numeric::{
            square::SquareLayouter,
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
    pub struct SquareLayerConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    pub struct SquareLayerCircuit<F: PrimeField> {
        pub input1: FieldTensor<F>,
    }

    impl<F: PrimeField> SquareLayerCircuit<F> {
        pub fn construct(input1: FieldTensor<F>) -> Self {
            Self { input1 }
        }
    }

    impl<F: PrimeField> Assign<F> for SquareLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for SquareLayerCircuit<F> {
        type Config = SquareLayerConfig<F>;
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
            let numeric_config = SquareLayouter::<F>::configure(meta, numeric_config);

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
            // Construct square layer chip
            let config_rc = config.numeric_config.clone();
            let square_layouter = SquareLayouter::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input1 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &square_layouter.config.columns,
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
                || "square layer",
                |mut region| {
                    Ok(square_layouter
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
    fn test_square_circuit() {
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
        // let input1:Vec<Int> = vec![1,2,3,4,5,6,7,8];
        // let expected_output:Vec<Int> = vec![2,6,12,20,30,42,56,72];

        let input1: Vec<Int> = vec![1, 2, 3];
        let expected_output: Vec<Int> = vec![1, 4, 9];

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
        let circuit = SquareLayerCircuit::construct(input1);

        // 运行电路验证
        let prover = MockProver::run(k as u32, &circuit, vec![expected_output]).unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }
}
