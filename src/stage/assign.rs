use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Column, Error},
};
use ndarray::{Array, IxDyn};

use crate::{
    numeric::NumericConfig,
    utils::{
        helpers::{to_field, AssignedTensor, CellRc, FieldTensor},
        math::Int,
    },
};

pub trait Assign<F: PrimeField> {
    fn assign_tensor_map(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensors_map: &HashMap<String, FieldTensor<F>>,
    ) -> Result<HashMap<String, AssignedTensor<F>>, Error> {
        Ok(layouter.assign_region(
            || "assign_tensor_map",
            |mut region| {
                let mut cell_idx = 0;
                let assigned_tensors = tensors_map
                    .iter()
                    .map(|(key, tensor)| {
                        let assigned_tensor = tensor
                            .iter()
                            .map(|cell| {
                                let row_idx = cell_idx / columns.len();
                                let col_idx = cell_idx % columns.len();
                                cell_idx += 1;
                                Ok(Rc::new(region.assign_advice(
                                    || "assign tensor cell",
                                    columns[col_idx],
                                    row_idx,
                                    || Value::known(*cell),
                                )?))
                            })
                            .collect::<Result<Vec<_>, Error>>()?;
                        Ok((
                            key.clone(),
                            match Array::from_shape_vec(IxDyn(tensor.shape()), assigned_tensor) {
                                Ok(x) => x,
                                Err(e) => panic!(
                                    "Error occurs at ForwardCircuit.assign_tensors_map: {:?}",
                                    e
                                ),
                            },
                        ))
                    })
                    .collect::<Result<HashMap<_, _>, Error>>()?;
                Ok(assigned_tensors)
            },
        )?)
    }

    fn assign_vector(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        input: &Vec<F>,
    ) -> Result<Vec<CellRc<F>>, Error> {
        Ok(layouter.assign_region(
            || "assign input",
            |mut region| {
                let mut cell_idx = 0;
                input
                    .iter()
                    .map(|cell| {
                        let row_idx = cell_idx / columns.len();
                        let col_idx = cell_idx % columns.len();
                        cell_idx += 1;
                        let out = region.assign_advice(
                            || "assign tensor cell",
                            columns[col_idx],
                            row_idx,
                            || Value::known(*cell),
                        )?;
                        Ok(Rc::new(out))
                    })
                    .collect::<Result<Vec<_>, Error>>()
            },
        )?)
    }

    fn assign_tensor(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensor: &FieldTensor<F>,
    ) -> Result<AssignedTensor<F>, Error> {
        Ok(layouter.assign_region(
            || "assign_tensors",
            |mut region| {
                let mut cell_idx = 0;
                match Array::from_shape_vec(
                    IxDyn(tensor.shape()),
                    tensor
                        .iter()
                        .map(|cell| {
                            let row_idx = cell_idx / columns.len();
                            let col_idx = cell_idx % columns.len();
                            cell_idx += 1;
                            Ok(Rc::new(region.assign_advice(
                                || "assign tensor cell",
                                columns[col_idx],
                                row_idx,
                                || Value::known(*cell),
                            )?))
                        })
                        .collect::<Result<Vec<_>, Error>>()?,
                ) {
                    Ok(x) => Ok(x),
                    Err(e) => panic!(
                        "Error occurs at FullyConnectedCircuit.assign_tensor: {:?}",
                        e
                    ),
                }
            },
        )?)
    }

    // For some hal2 horizon bug, we need to assign constants in two steps
    fn assign_constants(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<NumericConfig>,
    ) -> Result<HashMap<Int, CellRc<F>>, Error> {
        let sf = config.scale_factor;
        let bs = config.batch_size;
        let lr = config.learning_rate;
        // let min_val = config.min_val;
        // let max_val = config.max_val;

        let fixed_constants = layouter.assign_region(
            || "constants",
            |mut region| {
                let mut constants: HashMap<Int, CellRc<F>> = HashMap::new();

                let vals = vec![
                    0 as Int, 1, sf as Int, bs as Int, lr as Int, /*min_val, max_val*/
                ];
                for (i, val) in vals.iter().enumerate() {
                    let cell = region.assign_fixed(
                        || format!("constant_{}", i),
                        config.constants[0],
                        i,
                        || Value::known(to_field::<F>(*val)),
                    )?;
                    constants.insert(*val, Rc::new(cell));
                }

                Ok(constants)
            },
        )?;

        let advice_constants = layouter.assign_region(
            || "advice_constants",
            |mut region| {
                let mut constants: HashMap<Int, CellRc<F>> = HashMap::new();

                let vals = vec![0 as Int, 1, sf as Int, bs as Int, lr as Int];
                for (i, val) in vals.iter().enumerate() {
                    let row_idx = i / config.columns.len();
                    let col_idx = i % config.columns.len();
                    let cell = region.assign_advice(
                        || format!("constant_{}", i),
                        config.columns[col_idx],
                        row_idx,
                        || Value::known(to_field::<F>(*val)),
                    )?;
                    constants.insert(*val, Rc::new(cell));
                }

                for (k, v) in fixed_constants.iter() {
                    let v2 = constants.get(k).unwrap();
                    region.constrain_equal(v.cell(), v2.cell()).unwrap();
                }

                Ok(constants)
            },
        )?;

        Ok(advice_constants)
    }
}
