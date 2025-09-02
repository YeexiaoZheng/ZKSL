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

type DivSameConfig = NumericConfig;

pub struct DivSameLayouter<F: PrimeField> {
    pub config: Rc<DivSameConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DivSameLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> DivSameConfig {
        let selector = meta.complex_selector();
        let columns = &numeric_config.columns;
        let sf = Expression::Constant(F::from(numeric_config.scale_factor));
        let one = Expression::Constant(F::ONE);

        /* [8, 7, 6, 5] / 2 = [4, 3, 3, 2]
            advice      |  selector
            -----------------------
            inp        d|   1
            output     f|   0           // f means symbol of div, value of 1 or -1

            e.g.
            len_col = 6
            8  7   6   5   00   2 |  1
            4  3   3   2   00   1 |  0
        */
        // Assume that a / b = c
        let natural_lookup = numeric_config.tables.get(&NumericType::RowLookUp).unwrap()[0];
        // Check f * b >= 0
        meta.lookup("div same lookup0", |meta| {
            let s = meta.query_selector(selector);
            let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
            let f = meta.query_advice(columns[columns.len() - 1], Rotation::next());
            vec![(s.clone() * f * b, natural_lookup)]
        });
        for i in 0..(columns.len() - 1) {
            // Check |a * sf| - |b * c| >= 0
            // it contains
            // when a > 0, a * sf - b * c >= 0
            // when a < 0, a * sf - b * c <= 0
            // so we can use a * (a * sf - b * c) >= 0
            meta.lookup("div same lookup1", |meta| {
                let s = meta.query_selector(selector);
                let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
                let a = meta.query_advice(columns[i], Rotation::cur());
                let c = meta.query_advice(columns[i], Rotation::next());
                vec![(
                    s.clone() * (a.clone() * (a * sf.clone() - b * c)),
                    natural_lookup,
                )]
            });
            // Check f * (b * (c + 1) - a * sf) - 1 >= 0
            // f is the symbol of div, value of 1 or -1
            meta.lookup("div same lookup2", |meta| {
                let s = meta.query_selector(selector);
                let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
                let a = meta.query_advice(columns[i], Rotation::cur());
                let f = meta.query_advice(columns[columns.len() - 1], Rotation::next());
                let c = meta.query_advice(columns[i], Rotation::next());
                vec![(
                    s.clone() * (f * (b * (c + one.clone()) - a * sf.clone()) - one.clone()),
                    natural_lookup,
                )]
            });
            // // Check |a| - |b * c| >= 0
            // // it contains
            // // when a > 0, a - b * c >= 0
            // // when a < 0, a - b * c <= 0
            // // so we can use a * (a - b * c) >= 0
            // meta.lookup("div same lookup1", |meta| {
            //     let s = meta.query_selector(selector);
            //     let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
            //     let a = meta.query_advice(columns[i], Rotation::cur());
            //     let c = meta.query_advice(columns[i], Rotation::next());
            //     vec![(s.clone() * (a.clone() * (a - b * c)), natural_lookup)]
            // });
            // // Check f * (b * (c + 1) - a) - 1 >= 0
            // // f is the symbol of div, value of 1 or -1
            // meta.lookup("div same lookup2", |meta| {
            //     let s = meta.query_selector(selector);
            //     let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
            //     let a = meta.query_advice(columns[i], Rotation::cur());
            //     let f = meta.query_advice(columns[columns.len() - 1], Rotation::next());
            //     let c = meta.query_advice(columns[i], Rotation::next());
            //     vec![(
            //         s.clone() * (f * (b * (c + one.clone()) - a) - one.clone()),
            //         natural_lookup,
            //     )]
            // });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::DivSame, selector);

        DivSameConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for DivSameLayouter<F> {
    fn name(&self) -> String {
        "DivSame".to_string()
    }

    fn num_rows_per_unit(&self) -> usize {
        2
    }

    fn num_cols_per_row(&self) -> usize {
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
        // Check shapes
        assert_eq!(inputs[1].len(), 1);
        let div = inputs[1][0];
        let columns = &self.config.columns.clone();
        let sf = self.config.scale_factor as Int;

        // Assign input
        let inp = self.assign_row(
            region,
            columns,
            copy_advice,
            row_offset,
            &inputs[0],
            Some(Self::ZERO),
        )?;

        // Assign div and flag
        div.copy_advice(|| "", region, columns[columns.len() - 1], row_offset)?;
        let flag = region.assign_advice(
            || "",
            columns[columns.len() - 1],
            row_offset,
            || {
                div.value().map(|v| {
                    if to_primitive(v) >= 0 {
                        F::ONE
                    } else {
                        -F::ONE
                    }
                })
            },
        )?;
        println!("flag: {:?}", flag);

        // Enable the selector
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::DivSame).unwrap();
            selector.enable(region, row_offset).unwrap();
        }

        let res = inp
        .iter()
        .enumerate()
        .map(|(idx, inp)| {
            // Calculate value of a / b, a need to multiply by scale factor first
            let inp = inp.value().map(|x| to_field::<F>(to_primitive(x) * sf));
            let res = inp
                .zip(div.value().copied())
                .map(|(a, b)| to_field(to_primitive(&a) / to_primitive(&b)));
            region
                .assign_advice(|| "", columns[idx], row_offset + 1, || res)
                .unwrap()
        })
        .collect::<Vec<_>>();

        inp.iter().zip(res.iter()).for_each(|(inp, res)| {
            inp.value().copied().zip(res.value().copied()).zip(div.value().copied()).zip(flag.value().copied()).map(|(((a, c), b), f)| {
                let a = to_primitive(&a);
                let b = to_primitive(&b);
                let c = to_primitive(&c);
                let f = to_primitive(&f);
                println!("{} / {} = {}, flag: {}", a, b, c, f);
                println!("f * (b * (c + 1) - a) - 1: {}", f * (b * (c + 1) - a * sf) - 1);
            });
        });

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
        // Check input and weight shapes
        let input1 = inputs[0].clone();
        let input2 = inputs[1].clone();
        assert_eq!(input2.len(), 1);
        let original_input_len = input1.len();
        let cols_per_row = self.num_cols_per_row();

        // Calculate the number of units needed
        let used_units = self.used_units(input1.len());

        let output = input1
            .chunks(cols_per_row)
            .enumerate()
            .map(|(i, inp1)| {
                self.layout_unit(
                    region,
                    row_offset + i * rows_per_unit,
                    copy_advice,
                    &vec![inp1.to_vec(), input2.to_vec()],
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
            div_same::DivSameLayouter, lookups::row_lookup::RowLookUp, NumericConfig,
            NumericLayout, NumericType,
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
    pub struct DivSameLayerConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    pub struct DivSameLayerCircuit<F: PrimeField> {
        pub input1: FieldTensor<F>,
        pub input2: FieldTensor<F>,
    }

    impl<F: PrimeField> DivSameLayerCircuit<F> {
        pub fn construct(input1: FieldTensor<F>, input2: FieldTensor<F>) -> Self {
            Self { input1, input2 }
        }
    }

    impl<F: PrimeField> Assign<F> for DivSameLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for DivSameLayerCircuit<F> {
        type Config = DivSameLayerConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

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
            let numeric_config = RowLookUp::<F>::configure(meta, numeric_config);
            let numeric_config = DivSameLayouter::<F>::configure(meta, numeric_config);

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
            let div_same = DivSameLayouter::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input1 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &div_same.config.columns,
                &self.input1,
            )?;
            let input1 = input1
                .slice(s![..])
                .into_iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>();

            let input2 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs2"),
                &div_same.config.columns,
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
                NumericType::RowLookUp,
                layouter.namespace(|| "load lookup"),
            )
            .unwrap();

            println!("start compute.....");

            // Forward pass
            let mut outputs = vec![];
            outputs.extend(layouter.assign_region(
                || "div layer",
                |mut region| {
                    Ok(div_same
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
        let scale_factor = 100;
        let k = 20;
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
        let sample = num_cols - 1;
        // let input1: Vec<Int> = (0..sample).map(|_| random::<Int>() % 100 + 1).collect();
        let input2: Vec<Int> = vec![10];
        let input1: Vec<Int> = vec![0, 1, -2, 3, -4, 5, -6];
        // let input2: Vec<Int> = vec![2];
        let b = input2[0];
        // let expected_output: Vec<Int> = vec![1, 1, 0, -5, 6, -3, 4];
        let expected_output = input1
            .iter()
            .map(|a| a * scale_factor as Int / b)
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
        let circuit = DivSameLayerCircuit::construct(input1, input2);

        // 运行电路验证
        let prover = MockProver::run(k as u32, &circuit, vec![expected_output]).unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }
}
