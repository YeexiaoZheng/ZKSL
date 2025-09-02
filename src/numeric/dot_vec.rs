use std::{marker::PhantomData, rc::Rc};

use super::{NumericConfig, NumericLayout, NumericType};
use crate::numeric::sum::SumLayouter;
use halo2_proofs::plonk::Expression;
use halo2_proofs::{
    circuit::{AssignedCell, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};

type DotVecConfig = NumericConfig;

pub struct DotVecLayouter<F: PrimeField> {
    pub config: Rc<DotVecConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DotVecLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> DotVecConfig {
        let selector = meta.complex_selector();
        let columns = &numeric_config.columns;

        /* [1, 2, 3, 4] · [5, 6, 7, 8] = sum([5, 12, 21, 32]) = 70

            advice      |  selector
            -----------------------
            inp1      ❌|   1
            inp2      ✅|   0

           e.g.
           len_col = 6
           1  2   3   4   00   00   ｜  1
           5  6   7   8   00   70   ｜  0

        */
        meta.create_gate("dot vec gate", |meta| {
            let s = meta.query_selector(selector);
            // let num_inputs = columns.len() - 1;

            let inp1 = columns[0..columns.len() - 1]
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::cur()))
                .collect::<Vec<_>>();

            let inp2 = columns[0..columns.len() - 1]
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::next()))
                .collect::<Vec<_>>();

            let outp = meta.query_advice(columns[columns.len() - 1], Rotation::next());

            let res = inp1
                .iter()
                .zip(inp2.iter())
                .map(|(a, b)| a.clone() * b.clone())
                .fold(Expression::Constant(F::ZERO), |acc, x| acc + x);

            vec![s.clone() * (outp - res)]
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::DotVec, selector);

        DotVecConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for DotVecLayouter<F> {
    fn name(&self) -> String {
        "DotVec".to_string()
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
        assert_eq!(
            copy_advice, true,
            "dot vec layouter only supports copy_advice=true"
        );
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
            let selector = self.config.selectors.get(&NumericType::DotVec).unwrap();
            selector.enable(region, row_offset).unwrap();
        }

        // Compute the dot product, need to divide by the scale factor
        let value = inp1
            .iter()
            .zip(inp2.iter())
            .map(|(a, b)| a.value().copied() * b.value().copied())
            .fold(Value::known(F::ZERO), |acc, x| acc + x);

        // Assign the dot output
        let res =
            region.assign_advice(|| "", columns[columns.len() - 1], row_offset + 1, || value)?;

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
        assert_eq!(
            copy_advice, true,
            "DotVec layouter only supports copy_advice=true"
        );

        // Check inputs need to be even
        assert_eq!(inputs.len() % 2, 0);

        let mut row_offset = row_offset;

        let outputs = inputs
            .chunks(2)
            .map(|inputs| {
                // Check shapes
                let input1 = inputs[0].clone();
                let input2 = inputs[1].clone();
                assert_eq!(input1.len(), input2.len());
                let cols_per_row = self.num_cols_per_row();

                let used_units = self.used_units(input1.len());

                let outputs = input1
                    .chunks(cols_per_row)
                    .zip(input2.chunks(cols_per_row).enumerate())
                    .map(|(inp1, (i, inp2))| {
                        let output = self
                            .layout_unit(
                                region,
                                row_offset + i * rows_per_unit,
                                copy_advice,
                                &vec![inp1.to_vec(), inp2.to_vec()],
                                constants,
                            )
                            .unwrap();

                        output.first().unwrap().clone()
                    })
                    .collect::<Vec<_>>();

                row_offset = row_offset + used_units * rows_per_unit;

                let sum = SumLayouter::<F>::construct(self.config.clone());
                let sum_out = sum
                    .layout_customise(
                        region,
                        row_offset,
                        sum.num_rows_per_unit(),
                        copy_advice,
                        &vec![outputs.iter().collect::<Vec<_>>()],
                        constants,
                    )
                    .unwrap();
                row_offset = sum_out.1;
                sum_out.0.first().unwrap().clone()
            })
            .collect::<Vec<_>>();

        Ok((outputs, row_offset))
    }
}

#[cfg(test)]
mod tests {
    use std::{marker::PhantomData, rc::Rc, vec};

    use crate::{
        numeric::{dot_vec::DotVecLayouter, NumericConfig, NumericLayout},
        prover::prover::ProverKZG,
        stage::assign::Assign,
        utils::{
            helpers::{configure_static, get_circuit_numeric_config, to_field, FieldTensor},
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
    pub struct DotVecLayerConfig<F: PrimeField> {
        pub numeric_config: Rc<NumericConfig>,
        pub public: Column<Instance>,
        pub _marker: PhantomData<F>,
    }

    #[derive(Clone, Debug, Default)]
    pub struct DotVecLayerCircuit<F: PrimeField> {
        pub input1: FieldTensor<F>,
        pub input2: FieldTensor<F>,
    }

    impl<F: PrimeField> DotVecLayerCircuit<F> {
        pub fn construct(input1: FieldTensor<F>, input2: FieldTensor<F>) -> Self {
            Self { input1, input2 }
        }
    }

    impl<F: PrimeField> Assign<F> for DotVecLayerCircuit<F> {}

    impl<F: PrimeField> Circuit<F> for DotVecLayerCircuit<F> {
        type Config = DotVecLayerConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Default::default()
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
            let numeric_config = DotVecLayouter::<F>::configure(meta, numeric_config);

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
            // Construct dot layer chip
            let config_rc = config.numeric_config.clone();
            let dot_vec_layouter = DotVecLayouter::<F>::construct(config_rc.clone());

            // Assign input tensors
            let input1 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs"),
                &dot_vec_layouter.config.columns,
                &self.input1,
            )?;
            let input1 = input1
                .slice(s![..])
                .into_iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>();

            let input2 = self.assign_tensor(
                layouter.namespace(|| "assign_inputs2"),
                &dot_vec_layouter.config.columns,
                &self.input2,
            )?;
            let input2 = input2
                .slice(s![..])
                .into_iter()
                .map(|x| x.as_ref())
                .collect::<Vec<_>>();

            let input = &vec![input1.clone(), input2.clone(), input1, input2];

            // Assign constants
            let constants = self
                .assign_constants(layouter.namespace(|| "assign_constants"), config_rc.clone())
                .unwrap();
            let zero = constants.get(&0).unwrap().clone();
            let one = constants.get(&1).unwrap().clone();
            let constants = vec![zero.as_ref(), one.as_ref()];

            println!("start compute.....");

            // Forward pass
            let mut outputs = vec![];
            outputs.extend(layouter.assign_region(
                || "dotvec layer",
                |mut region| {
                    Ok(dot_vec_layouter
                        .layout(&mut region, 0, input, &constants)
                        .unwrap()
                        .0)
                },
            )?);

            // println!("outputs: {:?}", outputs);

            // Constrain public output
            let mut public_layouter = layouter.namespace(|| "public");
            for (i, cell) in outputs.iter().enumerate() {
                public_layouter.constrain_instance(cell.cell(), config.public, i)?;
            }

            Ok(())
        }
    }

    #[test]
    fn test_dot_vec_circuit() {
        // Parameters
        let scale_factor = 512;
        let k = 8;
        let num_cols = 5;

        configure_static(NumericConfig {
            k,
            num_cols,
            scale_factor,
            batch_size: 1,
            ..Default::default()
        });

        type F = Fr;
        // 定义测试输入和输出
        let input1: Vec<Int> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let input2: Vec<Int> = vec![5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8];
        let expected_output: Vec<Int> = vec![210, 210];

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
        let circuit = DotVecLayerCircuit::construct(input1, input2);

        // 运行电路验证
        let prover = MockProver::run(k as u32, &circuit, vec![expected_output.clone()]).unwrap();

        assert_eq!(prover.verify(), Ok(()));

        let mut prover = ProverKZG::construct(k, circuit, None);
        prover.load();
        // let _proof = prover.prove(vec![expected_output], vec![]);
        // let res = prover.verify(proof.clone(), proof.commitment.clone());
        // println!("KZG DotVec Circuit Verified: {}", res);

        // use crate::prover::prover::ProverKZG;
        // let mut prover = ProverKZG::construct(k, circuit, None);
        // let proof = prover.prove(vec![expected_output], vec![]);
        // let res = prover.verify(proof.clone(), proof.commitment.clone());
        // println!("KZG Forward Circuit Verified: {}", res);
    }
}
