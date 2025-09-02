use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region},
    halo2curves::ff::PrimeField,
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};
use log::debug;

use crate::{
    numeric::{NumericConfig, NumericLayout, NumericType},
    utils::math::{relu, Int},
};

use super::NonLinearNumericLayout;

const COLS_PER_UNIT: usize = 6;

pub struct GatherLookUp<F: PrimeField> {
    pub config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> GatherLookUp<F> {
    pub fn construct(config: Rc<NumericConfig>) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> NumericConfig {
        let numeric_type = NumericType::Gather;
        let selector = meta.complex_selector();
        let columns = numeric_config.columns;

        let input_lookup = numeric_config.gather_lookup[0];
        let output_lookup = numeric_config.gather_lookup[1];
        let gather_selector = numeric_config.gather_selector.unwrap();

        let units = columns.len() / COLS_PER_UNIT;
        for idx in 0..units {
            let inp_offset = idx * COLS_PER_UNIT;
            let out_offset = idx * COLS_PER_UNIT + 1;
            meta.lookup_any("gather lookup", |meta| {
                let s = meta.query_selector(selector);
                let g = meta.query_selector(gather_selector);
                let input_col = meta.query_advice(columns[inp_offset], Rotation::cur());
                let output_col = meta.query_advice(columns[out_offset], Rotation::cur());
                let input_lookup = meta.query_advice(input_lookup, Rotation::cur());
                let output_lookup = meta.query_advice(output_lookup, Rotation::cur());

                vec![
                    (s.clone() * input_col, input_lookup * g.clone()),
                    (s.clone() * output_col, output_lookup * g.clone()),
                    (s.clone(), g.clone()),
                ]
            });
        }

        let mut selectors = numeric_config.selectors;
        selectors.insert(numeric_type, selector);

        let mut maps = numeric_config.maps;
        let non_linear_map = Self::generate_map(
            numeric_config.scale_factor,
            numeric_config.min_val,
            numeric_config.max_val,
        );
        maps.insert(numeric_type, non_linear_map);

        NumericConfig {
            columns,
            selectors,
            maps,
            ..numeric_config
        }
    }
}

impl<F: PrimeField> NonLinearNumericLayout<F> for GatherLookUp<F> {
    fn generate_map(_scale_factor: u64, min_val: Int, max_val: Int) -> BTreeMap<Int, Int> {
        (min_val..max_val)
            .map(|x| (x, relu(x)))
            .collect::<BTreeMap<_, _>>()
    }

    fn get_numeric_config(&self) -> Rc<NumericConfig> {
        self.config.clone()
    }

    fn get_numeric_type(&self) -> NumericType {
        NumericType::Gather
    }

    fn load_lookups(&self, mut _layouter: impl Layouter<F>) -> Result<(), Error> {
        debug!("GatherLookUp lookups will be reloaded by other function");
        Ok(())
    }
}

impl<F: PrimeField> NumericLayout<F> for GatherLookUp<F> {
    fn name(&self) -> String {
        "GatherLookup".to_string()
    }

    fn num_rows_per_unit(&self) -> usize {
        <Self as NonLinearNumericLayout<F>>::num_rows_per_unit()
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
        let columns = &self.get_numeric_config().columns;

        let units = self.num_cols_per_row();
        let inp_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT])
            .collect::<Vec<_>>();
        let outp_cols = (0..units)
            .map(|x| columns[x * COLS_PER_UNIT + 1])
            .collect::<Vec<_>>();

        // Enable the selector
        if self.get_numeric_config().use_selectors {
            let selector = self.get_selector();
            selector.enable(region, row_offset).unwrap();
        }

        // Assign input
        let _input = self.assign_row(
            region,
            &inp_cols,
            copy_advice,
            row_offset,
            &inputs[0],
            Some(Self::ZERO),
        )?;

        let _output = self.assign_row(
            region,
            &outp_cols,
            copy_advice,
            row_offset,
            &inputs[1],
            Some(Self::ZERO),
        )?;

        Ok(vec![])
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
        let input = inputs[0].clone();
        // let input_len = input.len();
        let cols_per_row = self.num_cols_per_row();

        // Calculate the number of units needed
        let used_rows = self.used_units(input.len());

        let chunks = inputs
            .iter()
            .map(|input| input.chunks(cols_per_row).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let _output = input
            .chunks(cols_per_row)
            .enumerate()
            .map(|(i, _)| {
                let row_offset = row_offset + i * rows_per_unit;
                let chunks = chunks.iter().map(|x| x[i].to_vec()).collect::<Vec<_>>();
                <Self as NumericLayout<F>>::layout_unit(
                    self,
                    region,
                    row_offset,
                    copy_advice,
                    &chunks,
                    constants,
                )
                .unwrap()
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok((vec![], row_offset + used_rows * rows_per_unit))
    }
}
