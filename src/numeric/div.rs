use std::{marker::PhantomData, rc::Rc};

use super::{NumericConfig, NumericLayout, NumericType};
use crate::utils::{
    helpers::{to_field, to_primitive},
    math::Int,
};
use halo2_proofs::{
    circuit::{AssignedCell, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error, Expression},
    poly::Rotation,
};

type DivConfig = NumericConfig;

const COLS_PER_UNIT: usize = 5;

// TODO: Div need to adapt scale factor style, like DivSame
pub struct DivLayouter<F: PrimeField> {
    pub config: Rc<DivConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DivLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, numeric_config: NumericConfig) -> DivConfig {
        let selector = meta.complex_selector();
        let columns = &numeric_config.columns;
        let one = Expression::Constant(F::ONE);
        // let max = Expression::Constant(F::from(numeric_config.num_rows as u64));

        /*  7 / 2 = 3 ... 1
                    advice            |  selector
            -------------------------------------
            inp1 inp2 output reminder symbol |     1

            e.g.
            len_col = 6
            7  2   3   1   1  00  |  1
        */
        // Assume that a / b = c ... d  f means symbol of b, value is 1 or -1
        let natural_lookup = numeric_config
            .tables
            .get(&NumericType::NaturalLookUp)
            .unwrap()[0];
        let units = columns.len() / COLS_PER_UNIT;

        // Check a = b * c + d
        meta.create_gate("div a = b * c + d", |meta| {
            let s = meta.query_selector(selector);
            let mut constraints = vec![];
            for i in 0..units {
                let offset = i * COLS_PER_UNIT;
                let a = meta.query_advice(columns[offset], Rotation::cur());
                let b = meta.query_advice(columns[offset + 1], Rotation::cur());
                let c = meta.query_advice(columns[offset + 2], Rotation::cur());
                let d = meta.query_advice(columns[offset + 3], Rotation::cur());

                constraints.push(s.clone() * (a - (b * c + d)));
            }
            constraints
        });
        for i in 0..units {
            let offset = i * COLS_PER_UNIT;
            // Check b * f > 0
            meta.lookup("div f * b > 0", |meta| {
                let s = meta.query_selector(selector);
                let b = meta.query_advice(columns[offset + 1], Rotation::cur());
                let f = meta.query_advice(columns[offset + 4], Rotation::cur());
                vec![(s.clone() * (f * b - one.clone()), natural_lookup)]
            });
            // Check |b| > |d| ===> f * b > |d| ===> f * b > d and f * b > -d
            meta.lookup("div f * b - d > 0", |meta| {
                let s = meta.query_selector(selector);
                let b = meta.query_advice(columns[offset + 1], Rotation::cur());
                let d = meta.query_advice(columns[offset + 3], Rotation::cur());
                let f = meta.query_advice(columns[offset + 4], Rotation::cur());
                vec![(s.clone() * (f * b - d - one.clone()), natural_lookup)]
            });
            meta.lookup("div f * b + d > 0", |meta| {
                let s = meta.query_selector(selector);
                let b = meta.query_advice(columns[offset + 1], Rotation::cur());
                let d = meta.query_advice(columns[offset + 3], Rotation::cur());
                let f = meta.query_advice(columns[offset + 4], Rotation::cur());
                vec![(s.clone() * (f * b + d - one.clone()), natural_lookup)]
            });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Div, selector);

        DivConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for DivLayouter<F> {
    fn name(&self) -> String {
        "Div".to_string()
    }

    fn num_rows_per_unit(&self) -> usize {
        1
    }

    fn num_cols_per_row(&self) -> usize {
        self.config.columns.len() / COLS_PER_UNIT
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

        let units = self.num_cols_per_row();
        let inp1_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT])
            .collect::<Vec<_>>();
        let inp2_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT + 1])
            .collect::<Vec<_>>();
        let outp_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT + 2])
            .collect::<Vec<_>>();
        let remd_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT + 3])
            .collect::<Vec<_>>();
        let symb_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT + 4])
            .collect::<Vec<_>>();
        // Assign inputs
        let inp1 = self.assign_row(
            region,
            &inp1_cols,
            copy_advice,
            row_offset,
            &inputs[0],
            Some(Self::ZERO),
        )?;
        let inp2 = self.assign_row(
            region,
            &inp2_cols,
            copy_advice,
            row_offset,
            &inputs[1],
            Some(Self::ONE),
        )?;

        // Enable the selector
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Div).unwrap();
            selector.enable(region, row_offset).unwrap();
        }

        // Calculate value of a / b and a % b,
        let div_mod = |a: &F, b: &F| {
            let a = to_primitive(a);
            let b = to_primitive(b);
            let _div: Int = a / b;
            let _mod: Int = a % b;
            (to_field::<F>(_div), to_field::<F>(_mod))
        };

        // Calculate value of a % b
        let _remd = inp1
            .iter()
            .zip(inp2.iter())
            .enumerate()
            .map(|(idx, (a, b))| {
                // Calculate value of a % b
                let res = a.value().zip(b.value()).map(|(a, b)| div_mod(a, b).1);
                region
                    .assign_advice(|| "", remd_cols[idx], row_offset, || res)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let _symbol = inp2
            .iter()
            .enumerate()
            .map(|(idx, b)| {
                region
                    .assign_advice(
                        || "",
                        symb_cols[idx],
                        row_offset,
                        || {
                            b.value().map(|v| {
                                if to_primitive(v) >= 0 {
                                    F::ONE
                                } else {
                                    -F::ONE
                                }
                            })
                        },
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        Ok(inp1
            .iter()
            .zip(inp2.iter())
            .enumerate()
            .map(|(idx, (a, b))| {
                // Calculate value of a / b
                let res = a.value().zip(b.value()).map(|(a, b)| div_mod(a, b).0);
                region
                    .assign_advice(|| "", outp_cols[idx], row_offset, || res)
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
        // Check input and weight shapes
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
        numeric::{
            div::DivLayouter, nonlinear::natural::NaturalLookUp, NumericConfig, NumericLayout,
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
    use rand::random;

    #[derive(Clone, Debug)]
    pub struct DivLayerConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    pub struct DivLayerCircuit<F: PrimeField> {
        pub input1: FieldTensor<F>,
        pub input2: FieldTensor<F>,
    }

    impl<F: PrimeField> DivLayerCircuit<F> {
        pub fn construct(input1: FieldTensor<F>, input2: FieldTensor<F>) -> Self {
            Self { input1, input2 }
        }
    }

    impl<F: PrimeField> Assign<F> for DivLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for DivLayerCircuit<F> {
        type Config = DivLayerConfig<F>;
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
            let numeric_config = DivLayouter::<F>::configure(meta, numeric_config);

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
            // Construct div layer chip
            let config_rc = config.numeric_config.clone();
            let div_layouter = DivLayouter::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input1 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &div_layouter.config.columns,
                &self.input1,
            )?;
            let input1 = input1
                .slice(s![..])
                .into_iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>();

            let input2 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs2"),
                &div_layouter.config.columns,
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
            let one = constants.get(&1).unwrap().clone();
            let constants = vec![zero.as_ref(), one.as_ref()];

            match_load_lookups(
                config.numeric_config,
                NumericType::NaturalLookUp,
                layouter.namespace(|| "load lookup"),
            )
            .unwrap();

            println!("start compute.....");

            // Forward pass
            let mut outputs = vec![];
            outputs.extend(layouter.assign_region(
                || "div layer",
                |mut region| {
                    Ok(div_layouter
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
    fn test_div_circuit() {
        // Parameters
        let scale_factor = 1024;
        let k = 14;
        let num_cols = 9;

        configure_static(NumericConfig {
            k,
            num_cols,
            scale_factor,
            use_selectors: true,
            ..Default::default()
        });

        type F = Fr;
        // 定义测试输入和输出
        let sample = num_cols;
        let input1: Vec<Int> = (0..sample).map(|_| random::<Int>() % 100 + 1).collect();
        let input2: Vec<Int> = (0..sample).map(|_| random::<Int>() % 100 + 1).collect();
        // let expected_output: Vec<Int> = vec![1, 1, 0, -5, 6, -3, 4];
        let expected_output = input1
            .iter()
            .zip(input2.iter())
            .map(|(a, b)| a / b)
            .collect::<Vec<_>>();

        println!("input1: {:?}", input1);
        println!("input2: {:?}", input2);
        println!("expected_output: {:?}", expected_output);

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
        let circuit = DivLayerCircuit::construct(input1, input2);

        // 运行电路验证
        let prover = MockProver::run(k as u32, &circuit, vec![expected_output]).unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }
}
