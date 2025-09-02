use std::{collections::BTreeMap, rc::Rc, vec};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Column, Error},
};
use ndarray::{Array, IxDyn};
use rand::{Rng, SeedableRng};

use crate::{
    numeric::NumericConfig,
    utils::{
        helpers::{to_field, to_primitive, AssignedTensor, CellRc, FieldTensor, ValueTensor},
        math::Int,
    },
};

pub trait Assign<F: PrimeField> {
    fn assign_tensor_map(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensors_map: &BTreeMap<String, FieldTensor<F>>,
    ) -> Result<BTreeMap<String, AssignedTensor<F>>, Error> {
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
                    .collect::<Result<BTreeMap<_, _>, Error>>()?;
                Ok(assigned_tensors)
            },
        )?)
    }

    fn copy_assign_tensor_map(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensors_map: &BTreeMap<String, AssignedTensor<F>>,
    ) -> Result<BTreeMap<String, AssignedTensor<F>>, Error> {
        Ok(layouter.assign_region(
            || "copy_assign_tensor_map",
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
                                    || cell.value().copied(),
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
                    .collect::<Result<BTreeMap<_, _>, Error>>()?;
                Ok(assigned_tensors)
            },
        )?)
    }

    // TOFIX: when columns.len() > 1, this is wrong
    fn copy_assign_vector(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        input: &Vec<&AssignedCell<F, F>>,
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
                        let out = cell.copy_advice(
                            || "copy assign",
                            &mut region,
                            columns[col_idx],
                            row_idx,
                        )?;
                        Ok(Rc::new(out))
                    })
                    .collect::<Result<Vec<_>, Error>>()
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

    fn assign_value_tensor(
        &self,
        mut layouter: impl Layouter<F>,
        columns: &Vec<Column<Advice>>,
        tensor: &ValueTensor<F>,
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
                                || *cell,
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

    // For some halo2 horizon bug, we need to assign constants in two steps
    fn assign_constants(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<NumericConfig>,
    ) -> Result<BTreeMap<Int, CellRc<F>>, Error> {
        let sf = config.scale_factor;
        let bs = config.batch_size;
        let rlr = config.reciprocal_learning_rate;
        let fnum = config.feature_num;
        // let min_val = config.min_val;
        // let max_val = config.max_val;

        let fixed_constants = layouter.assign_region(
            || "constants",
            |mut region| {
                let mut constants: BTreeMap<Int, CellRc<F>> = BTreeMap::new();
                // TODO: fnum need to be assigned at other position
                let vals = vec![
                    0 as Int,
                    1,
                    2,
                    sf as Int,
                    bs as Int,
                    rlr as Int,
                    fnum as Int,
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
                let mut constants: BTreeMap<Int, CellRc<F>> = BTreeMap::new();
                // TODO: fnum need to be assigned at other position
                let vals = vec![
                    0 as Int,
                    1,
                    2,
                    sf as Int,
                    bs as Int,
                    rlr as Int,
                    fnum as Int,
                ];
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

    fn assign_other_constants(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<NumericConfig>,
        map: &BTreeMap<Int, CellRc<F>>,
        constants: Vec<F>,
    ) -> Result<BTreeMap<Int, CellRc<F>>, Error> {
        // select constants that are not in the map
        let vals = constants
            .into_iter()
            .filter(|v| !map.contains_key(&to_primitive(v)))
            .map(|v| to_primitive(&v))
            .collect::<Vec<_>>();

        let fixed = layouter.assign_region(
            || "constants",
            |mut region| {
                let mut fixed = vec![];
                for (i, val) in vals.iter().enumerate() {
                    let cell = region.assign_fixed(
                        || format!("constant_{}", i),
                        config.constants[0],
                        i,
                        || Value::known(to_field::<F>(*val)),
                    )?;
                    fixed.push(Rc::new(cell));
                }
                Ok(fixed)
            },
        )?;

        let advice_constants = layouter.assign_region(
            || "advice_other_constants",
            |mut region| {
                let mut map = map.clone();

                for (i, val) in vals.iter().enumerate() {
                    let row_idx = i / config.columns.len();
                    let col_idx = i % config.columns.len();
                    let cell = region.assign_advice(
                        || format!("constant_{}", i),
                        config.columns[col_idx],
                        row_idx,
                        || Value::known(to_field::<F>(*val)),
                    )?;
                    region
                        .constrain_equal(cell.cell(), fixed[i].cell())
                        .unwrap();
                    map.insert(*val, Rc::new(cell));
                }

                Ok(map)
            },
        )?;

        Ok(advice_constants)
    }

    fn assign_random(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<NumericConfig>,
    ) -> Result<Vec<CellRc<F>>, Error> {
        let seed = [0; 32];
        let mut rng = rand::rngs::StdRng::from_seed(seed);
        let random = (0..config.random_size)
            .map(|_| rng.gen_range(0..config.random_size as Int))
            .collect::<Vec<Int>>();
        let random = random.into_iter().map(|x| to_field(x)).collect::<Vec<_>>();

        // let fixed = layouter.assign_region(
        //     || "constants",
        //     |mut region| {
        //         let mut fixed = vec![];
        //         for (i, val) in random.iter().enumerate() {
        //             let cell = region.assign_fixed(
        //                 || format!("constant_{}", i),
        //                 config.constants[0],
        //                 i,
        //                 || Value::known(*val),
        //             )?;
        //             fixed.push(Rc::new(cell));
        //         }
        //         Ok(fixed)
        //     },
        // )?;

        let advice_random = layouter.assign_region(
            || "advice random",
            |mut region| {
                let mut r = vec![];

                for (i, val) in random.iter().enumerate() {
                    let row_idx = i / config.columns.len();
                    let col_idx = i % config.columns.len();
                    let cell = region.assign_advice(
                        || format!("random_{}", i),
                        config.columns[col_idx],
                        row_idx,
                        || Value::known(*val),
                    )?;
                    // region
                    //     .constrain_equal(cell.cell(), fixed[i].cell())
                    //     .unwrap();
                    r.push(Rc::new(cell));
                }

                Ok(r)
            },
        )?;

        Ok(advice_random)
    }

    fn assign_random_by_vec(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<NumericConfig>,
        random: Vec<F>,
    ) -> Result<Vec<CellRc<F>>, Error> {
        let advice_random = layouter.assign_region(
            || "advice random",
            |mut region| {
                let mut r = vec![];

                for (i, val) in random.iter().enumerate() {
                    let row_idx = i / config.columns.len();
                    let col_idx = i % config.columns.len();
                    let cell = region.assign_advice(
                        || format!("random_{}", i),
                        config.columns[col_idx],
                        row_idx,
                        || Value::known(*val),
                    )?;
                    r.push(Rc::new(cell));
                }

                Ok(r)
            },
        )?;

        Ok(advice_random)
    }

    fn dot_vector_f(&self, input1: Vec<F>, input2: Vec<F>) -> F {
        let mut sum = F::ZERO;
        for (i, x) in input1.iter().enumerate() {
            sum += *x * input2[i];
        }
        sum
    }

    fn assign_gather_lookup(
        &self,
        region: &mut Region<F>,
        config: Rc<NumericConfig>,
        embeddings: Vec<Vec<AssignedCell<F, F>>>,
    ) -> Result<(), Error> {
        let input_lookup = config.gather_lookup[0];
        let output_lookup = config.gather_lookup[1];

        let mut offset = 0;
        for (_, emb) in embeddings.iter().enumerate() {
            // println!("embedding: {:?}", emb.len());
            for (j, e) in emb.iter().enumerate() {
                offset += 1;
                config.gather_selector.unwrap().enable(region, offset)?;
                region.assign_advice(
                    || "gather input table",
                    input_lookup,
                    offset,
                    || Value::known(to_field::<F>(j as Int)),
                )?;

                e.copy_advice(|| "gather output table", region, output_lookup, offset)?;
            }
        }
        Ok(())
    }
}
