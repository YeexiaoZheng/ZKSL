use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};

use super::{NumericConfig, NumericLayout, NumericType};

type AddConfig = NumericConfig;

pub struct AddLayouter<F: PrimeField> {
    pub config: Rc<AddConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> AddLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, numeric_config: NumericConfig) -> AddConfig {
        let selector = meta.selector();
        let columns = &numeric_config.columns;

        /* [1, 2, 3, 4] + [5, 6, 7, 8] = [6, 8, 10, 12]
            advice      |  selector
            -----------------------
            inp1        |   0
            inp2        |   1
            output      |   0

            e.g.
            len_col = 5
            1  2   3   4   00   |  0
            5  6   7   8   00   |  1
            6  8   10  12  00   |  0
        */
        meta.create_gate("add gate", |meta| {
            let s = meta.query_selector(selector);
            (0..columns.len())
                .into_iter()
                .map(|i| {
                    let inp1 = meta.query_advice(columns[i], Rotation::prev());
                    let inp2 = meta.query_advice(columns[i], Rotation::cur());
                    let outp = meta.query_advice(columns[i], Rotation::next());
                    s.clone() * (inp1 + inp2 - outp)
                })
                .collect::<Vec<_>>()
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Add, selector);

        AddConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for AddLayouter<F> {
    fn name(&self) -> String {
        "Add".to_string()
    }

    fn num_rows_per_unit(&self) -> usize {
        3
    }

    fn num_cols_per_row(&self) -> usize {
        self.config.columns.len()
    }

    fn layout_unit(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        copy_advice: bool,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        _constants: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check shapes
        assert_eq!(inputs[0].len(), inputs[1].len());
        let columns = &self.config.columns.clone();

        // Assign inputs
        let inp1 = self.assign_row(
            region,
            columns,
            copy_advice,
            row_offset,
            &inputs[0],
            Some(Self::ZERO),
        )?;
        let inp2 = self.assign_row(
            region,
            columns,
            copy_advice,
            row_offset + 1,
            &inputs[1],
            Some(Self::ZERO),
        )?;

        // Enable the selector
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Add).unwrap();
            selector.enable(region, row_offset + 1).unwrap();
        }

        Ok(inp1
            .iter()
            .zip(inp2.iter())
            .enumerate()
            .map(|(idx, (a, b))| {
                // Calculate value of a + b
                let res = a.value().copied() + b.value().copied();
                region
                    .assign_advice(|| "", columns[idx], row_offset + 2, || res)
                    .unwrap()
            })
            .collect::<Vec<_>>())
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
        let input1 = inputs[0].clone();
        let input2 = inputs[1].clone();
        assert_eq!(input1.len(), input2.len());
        let original_input_len = input1.len();
        let cols_per_row = self.num_cols_per_row();

        // Calculate the number of units needed
        let used_units = self.used_units(input1.len());

        let output = input1
            .chunks(cols_per_row)
            .zip(input2.chunks(cols_per_row).enumerate())
            .map(|(inp1, (i, inp2))| {
                self.layout_unit(
                    region,
                    row_offset + i * rows_per_unit,
                    copy_advice,
                    &vec![inp1.to_vec(), inp2.to_vec()],
                    constants,
                )
                .unwrap()
            })
            .flatten()
            .collect::<Vec<_>>();

        let row_offset = row_offset + used_units * rows_per_unit;
        Ok((output[0..original_input_len].to_vec(), row_offset))
    }
}

#[cfg(test)]
mod tests {
    use std::{marker::PhantomData, rc::Rc, vec};

    use crate::{
        numeric::{add::AddLayouter, NumericConfig, NumericLayout},
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
    pub struct AddLayerConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    #[derive(Clone, Debug, Default)]
    pub struct AddLayerCircuit<F: PrimeField> {
        pub input1: FieldTensor<F>,
        pub input2: FieldTensor<F>,
    }

    impl<F: PrimeField> AddLayerCircuit<F> {
        pub fn construct(input1: FieldTensor<F>, input2: FieldTensor<F>) -> Self {
            Self { input1, input2 }
        }
    }

    impl<F: PrimeField> Assign<F> for AddLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for AddLayerCircuit<F> {
        type Config = AddLayerConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Default::default()
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
            let numeric_config = AddLayouter::<F>::configure(meta, numeric_config);

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
            // Construct add layer chip
            let config_rc = config.numeric_config.clone();
            let add_layouter = AddLayouter::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input1 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &add_layouter.config.columns,
                &self.input1,
            )?;
            let input1 = input1
                .slice(s![..])
                .into_iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>();

            let input2 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs2"),
                &add_layouter.config.columns,
                &self.input2,
            )?;
            let input2 = input2
                .slice(s![..])
                .into_iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>();

            let input = &vec![input1, input2];

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
                || "add layer",
                |mut region| {
                    Ok(add_layouter
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
    fn test_add_circuit() {
        // Parameters
        let scale_factor = 1024;
        let k = 14;
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
        let input2: Vec<Int> = vec![2, 3, 4, 5, 6, 7, 8, 9];
        let expected_output: Vec<Int> = vec![3, 5, 7, 9, 11, 13, 15, 17];

        // let input1:Vec<Int> = vec![1,2,3];
        // let input2:Vec<Int> = vec![2,3,4];
        // let expected_output:Vec<Int> = vec![3,5,7];

        let input1 = Array::from_shape_vec(
            IxDyn(&[input1.len()]),
            input1.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>(),
        )
        .unwrap();

        // println!("input1: {:?}", input1);

        let input2 = Array::from_shape_vec(
            IxDyn(&[input2.len()]),
            input2.iter().map(|x| to_field::<F>(*x)).collect::<Vec<_>>(),
        )
        .unwrap();

        // println!("input2: {:?}", input2);

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
        let circuit = AddLayerCircuit::construct(input1, input2);

        // 运行电路验证
        let prover = MockProver::run(k as u32, &circuit, vec![expected_output.clone()]).unwrap();

        assert_eq!(prover.verify(), Ok(()));

        // let mut prover = ProverKZG::construct(k, circuit, None);
        // let proof = prover.prove(vec![expected_output], vec![]);
        // let res = prover.verify(proof.clone(), proof.commitment.clone());
        // println!("KZG Forward Circuit Verified: {}", res);
    }
}
