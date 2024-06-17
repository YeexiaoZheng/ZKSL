use std::marker::PhantomData;

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Column, ConstraintSystem, Error, Expression},
    poly::Rotation,
};

use crate::layers::layer::{AssignedTensor, AssignedTensorRef};

use super::numeric::{Numeric, NumericConfig, NumericType, _NumericConfig};

type DotConfig = _NumericConfig;

pub struct DotChip<F: PrimeField> {
    pub config: DotConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> DotChip<F> {
    pub fn construct(config: DotConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: _NumericConfig,
    ) -> _NumericConfig {
        let selector = meta.selector();
        let columns = &numeric_config.columns;

        meta.create_gate("dot gate", |meta| {
            let s = meta.query_selector(selector);

            let num_inputs = (columns.len() - 1) / 2;
            let gate_inputs = columns[0..num_inputs]
                .to_vec()
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::cur()))
                .collect::<Vec<_>>();
            let gate_weights = columns[num_inputs..columns.len() - 1]
                .to_vec()
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::cur()))
                .collect::<Vec<_>>();
            let gate_outputs = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

            let res = gate_inputs
                .iter()
                .zip(gate_weights)
                .map(|(a, b)| a.clone() * b.clone())
                .fold(Expression::Constant(F::ZERO), |a, b| a + b);

            vec![s * (res - gate_outputs)]
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::Dot, vec![selector]);

        _NumericConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }

    pub fn get_input_columns(config: &_NumericConfig) -> Vec<Column<Advice>> {
        let num_inputs = (config.columns.len() - 1) / 2;
        config.columns[0..num_inputs].to_vec()
    }

    pub fn get_weight_columns(config: &_NumericConfig) -> Vec<Column<Advice>> {
        let num_inputs = (config.columns.len() - 1) / 2;
        config.columns[num_inputs..config.columns.len() - 1].to_vec()
    }

    fn op_row_region(
        &self,
        region: &mut Region<F>,
        row_offset: usize,
        inputs: &Vec<Vec<&AssignedCell<F, F>>>,
        zeros: &Vec<&AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Check input and weight shapes
        assert_eq!(inputs.len(), 2);
        let input = &inputs[0];
        let weight = &inputs[1];
        assert_eq!(input.len(), weight.len());
        assert_eq!(input.len(), self.num_input_cols_per_row());

        // Enable selectors
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::Dot).unwrap()[0];
            selector.enable(region, row_offset).unwrap();
        }

        // Assign input and weight columns by copy advice
        let inp_cols = DotChip::<F>::get_input_columns(&self.config);
        input
            .iter()
            .enumerate()
            .map(|(i, cell)| cell.copy_advice(|| "", region, inp_cols[i], row_offset))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let weight_cols = DotChip::<F>::get_weight_columns(&self.config);
        weight
            .iter()
            .enumerate()
            .map(|(i, cell)| cell.copy_advice(|| "", region, weight_cols[i], row_offset))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // All columns need to be assigned
        let zero = zeros[0];
        if self.config.columns.len() % 2 == 0 {
            zero.copy_advice(
                || "",
                region,
                self.config.columns[self.config.columns.len() - 2],
                row_offset,
            )
            .unwrap();
        }

        let res = region
            .assign_advice(
                || "",
                self.config.columns[self.config.columns.len() - 1],
                row_offset,
                || {
                    input
                        .iter()
                        .zip(weight.iter())
                        .map(|(a, b)| a.value().map(|x: &F| *x) * b.value())
                        .reduce(|a, b| a + b)
                        .unwrap()
                },
            )
            .unwrap();

        Ok(vec![res])
    }
}

impl<F: PrimeField> Numeric<F> for DotChip<F> {
    fn name(&self) -> String {
        "Dot".to_string()
    }

    fn num_cols_per_op(&self) -> usize {
        self.config.columns.len()
    }

    fn num_input_cols_per_row(&self) -> usize {
        self.config.columns.len() - 1
    }

    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
    ) -> Result<Vec<AssignedTensor<F>>, Error> {
        // Check input and weight shapes
        let input = inputs[0].clone();
        let weight = inputs[1].clone();
        assert_eq!(input.ndim(), 1);
        assert_eq!(weight.ndim(), 1);
        assert_eq!(input.shape(), weight.shape());

        // Turn input and weight into Vec
        let input = input.into_iter().collect::<Vec<_>>();
        let weight = weight.into_iter().collect::<Vec<_>>();

        // Assign input and weight columns
        let outputs = layouter
            .assign_region(
                || "dot rows",
                |mut region| {
                    let mut outputs = vec![];
                    for i in 0..inputs.len() / self.num_input_cols_per_row() {
                        let input = input[i * self.num_input_cols_per_row()
                            ..(i + 1) * self.num_input_cols_per_row()]
                            .to_vec();
                        let weight = weight[i * self.num_input_cols_per_row()
                            ..(i + 1) * self.num_input_cols_per_row()]
                            .to_vec();
                        let res = self
                            .op_row_region(
                                &mut region,
                                i,
                                &vec![input, weight],
                                &vec![zero.clone()],
                            ).unwrap();
                        outputs.push(res[0].clone());
                    }
                    Ok(outputs)
                },
            )?;
        println!("outputs len: {}", outputs.len());

        // let adder_chip = AdderChip::<F>::construct(self.config.clone());
        // let tmp = outputs.iter().map(|x| x).collect::<Vec<_>>();
        // Ok(adder_chip
        //     .forward(
        //         layouter.namespace(|| "dot prod adder"),
        //         &vec![tmp],
        //         single_inputs,
        //     )
        //     .unwrap())
        Ok(vec![])
    }
}
