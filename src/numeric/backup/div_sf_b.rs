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

type DivSFConfig = NumericConfig;

pub struct DivSFLayouter<F: PrimeField> {
    pub config: Rc<DivSFConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DivSFLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, numeric_config: NumericConfig) -> DivSFConfig {
        let selector = meta.complex_selector();
        let columns = &numeric_config.columns;
        let sf = Expression::Constant(to_field(numeric_config.scale_factor as Int));
        let one = Expression::Constant(F::ONE);

        /* [8, 7, 6, 5] / sf(2) = [4, 3, 3, 2]
            advice      |  selector
            -----------------------
            inp         |   0
            reminder    |   1
            output      |   0

            e.g.
            len_col = 5
            8  7   6   5   00   |  0
            0  1   0   1   00   |  1
            4  3   3   2   00   |  0
        */
        // Assume that a / b = c + r
        meta.create_gate("div sf", |meta| {
            (0..columns.len())
                .map(|i| {
                    let s = meta.query_selector(selector);
                    let inp = meta.query_advice(columns[i], Rotation::prev());
                    let r = meta.query_advice(columns[i], Rotation::cur());
                    let outp = meta.query_advice(columns[i], Rotation::next());
                    s.clone() * (inp.clone() - (sf.clone() * outp.clone()) - r.clone())
                })
                .collect::<Vec<_>>()
        });
        let natural_lookup = numeric_config
            .tables
            .get(&NumericType::NaturalLookUp)
            .unwrap()[0];
        for i in 0..columns.len() {
            // Check |b| - |r| > 0
            // We know b is sf, it always great than 0
            // So we only need to check b - r > 0 and b + r > 0
            // and b, r is integer,so we check b - r - 1 >= 0 and b + r - 1 >= 0
            meta.lookup("div sf lookup1", |meta| {
                let s = meta.query_selector(selector);
                let r = meta.query_advice(columns[i], Rotation::cur());
                vec![(
                    s.clone() * (sf.clone() - r.clone() - one.clone()),
                    natural_lookup,
                )]
            });
            meta.lookup("div sf lookup2", |meta| {
                let s = meta.query_selector(selector);
                let r = meta.query_advice(columns[i], Rotation::cur());
                vec![(
                    s.clone() * (sf.clone() + r.clone() - one.clone()),
                    natural_lookup,
                )]
            });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::DivSF, selector);

        DivSFConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for DivSFLayouter<F> {
    fn name(&self) -> String {
        "DivSF".to_string()
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

        // Enable the selector
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::DivSF).unwrap();
            selector.enable(region, row_offset + 1).unwrap();
        }

        // Assign reminder
        let _ = inp
            .iter()
            .enumerate()
            .map(|(idx, inp)| {
                // Calculate value of a % sf
                let res = inp
                    .value()
                    .copied()
                    .map(|a| to_field::<F>(to_primitive(&a) % sf));
                region
                    .assign_advice(|| "", columns[idx], row_offset + 1, || res)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        Ok(inp
            .iter()
            .enumerate()
            .map(|(idx, inp)| {
                // Calculate value of a / sf
                let res = inp
                    .value()
                    .copied()
                    .map(|a| to_field(to_primitive(&a) / sf));
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
        // Check input shapes
        let input1 = inputs[0].clone();
        let original_input_len = input1.len();
        let cols_per_row = self.num_cols_per_row();

        // Calculate the number of units needed
        let used_units = self.used_units(input1.len());

        let output = input1
            .chunks(cols_per_row)
            .enumerate()
            .map(|(i, inp)| {
                self.layout_unit(
                    region,
                    row_offset + i * rows_per_unit,
                    copy_advice,
                    &vec![inp.to_vec()],
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
            div_sf::DivSFLayouter, nonlinear::natural::NaturalLookUp, NumericConfig, NumericLayout,
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
    pub struct DivSFLayerConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    pub struct DivSFLayerCircuit<F: PrimeField> {
        pub input1: FieldTensor<F>,
    }

    impl<F: PrimeField> DivSFLayerCircuit<F> {
        pub fn construct(input1: FieldTensor<F>) -> Self {
            Self { input1 }
        }
    }

    impl<F: PrimeField> Assign<F> for DivSFLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for DivSFLayerCircuit<F> {
        type Config = DivSFLayerConfig<F>;
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
            let numeric_config = NaturalLookUp::<F>::configure(meta, numeric_config);
            let numeric_config = DivSFLayouter::<F>::configure(meta, numeric_config);

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
            let div_sf = DivSFLayouter::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input1 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &div_sf.config.columns,
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
            let one = constants.get(&1).unwrap().clone();
            let constants = vec![zero.as_ref(), one.as_ref()];

            match_load_lookups(
                config.numeric_config,
                NumericType::NaturalLookUp,
                layouter.namespace(|| "load_lookups"),
            )
            .unwrap();

            println!("start compute.....");

            // Forward pass
            let mut outputs = vec![];
            outputs.extend(layouter.assign_region(
                || "div layer",
                |mut region| Ok(div_sf.layout(&mut region, 0, input, &constants).unwrap().0),
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
    fn test_div_sf_circuit() {
        // Parameters
        let scale_factor = 2;
        let k = 18;
        let num_cols = 4;
        let lr = 1;

        configure_static(NumericConfig {
            k,
            num_cols,
            scale_factor,
            batch_size: 1,
            learning_rate: lr,
            use_selectors: true,
            ..Default::default()
        });

        type F = Fr;
        // 定义测试输入和输出
        let input1: Vec<Int> = vec![5, 6, -7, -8];
        let expected_output: Vec<Int> = input1.iter().map(|x| x / scale_factor as Int).collect();

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
        let circuit = DivSFLayerCircuit::construct(input1);

        // 运行电路验证
        let prover = MockProver::run(k as u32, &circuit, vec![expected_output]).unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }
}
