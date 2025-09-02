use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};

use super::{NumericConfig, NumericLayout, NumericType};

type MulSameConfig = NumericConfig;

pub struct MulSameLayouter<F: PrimeField> {
    pub config: Rc<MulSameConfig>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> MulSameLayouter<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> MulSameConfig {
        let selector = meta.selector();
        let columns = &numeric_config.columns;

        /* [1, 2, 1, 2] * 2 = [2, 4, 2, 4]
            advice      |  selector
            -----------------------
            inp1       m|   1
            output      |   0

            e.g.
            len_col = 6
            1  2   1   2   00   2 |  1
            2  4   2   4   00  00 |  0
        */
        // TODO: still bug
        meta.create_gate("mul same gate", |meta| {
            let s = meta.query_selector(selector);
            let mul = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
            (0..(columns.len() - 1))
                .into_iter()
                .map(|i| {
                    let inp = meta.query_advice(columns[i], Rotation::cur());
                    let outp = meta.query_advice(columns[i], Rotation::next());
                    s.clone() * (inp * mul.clone() - outp)
                })
                .collect::<Vec<_>>()
        });

        let mut selectors = numeric_config.selectors;
        selectors.insert(NumericType::MulSame, selector);

        MulSameConfig {
            columns: numeric_config.columns,
            selectors,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NumericLayout<F> for MulSameLayouter<F> {
    fn name(&self) -> String {
        "MulSame".to_string()
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
        // Check shapes
        assert_eq!(inputs[1].len(), 1);
        let mul = inputs[1][0];
        let columns = &self.config.columns.clone();

        // Assign input
        let inp = self.assign_row(
            region,
            columns,
            copy_advice,
            row_offset,
            &inputs[0],
            Some(Self::ZERO),
        )?;

        // Assign mul
        mul.copy_advice(|| "", region, columns[columns.len() - 1], row_offset)?;

        // Enable the selector
        if self.config.use_selectors {
            let selector = self.config.selectors.get(&NumericType::MulSame).unwrap();
            selector.enable(region, row_offset).unwrap();
        }

        Ok(inp
            .iter()
            .enumerate()
            .map(|(idx, inp)| {
                // Calculate value of a * b
                let res = inp.value().copied() * mul.value().copied();
                region
                    .assign_advice(|| "", columns[idx], row_offset + 1, || res)
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
