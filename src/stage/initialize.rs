use std::{
    collections::{BTreeSet, HashMap},
    rc::Rc,
};

use halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
    plonk::{Advice, Column, Error, ErrorFront},
};
use ndarray::{Array, IxDyn, ShapeError};

use crate::{
    graph::Graph,
    numerics::numeric::{NumericConfig, NumericType},
    utils::{
        helpers::{to_field, AssignedTensor, CellRc, FieldTensor, Tensor},
        matcher::{match_consumer, match_op_type},
        math::Int,
    },
};

pub trait Initialize<F: PrimeField> {
    fn initialize(graph: Graph) -> (BTreeSet<NumericType>, HashMap<String, FieldTensor<F>>) {
        let mut used_numerics = BTreeSet::new();
        for node in graph.nodes.iter() {
            let op_type = match_op_type(node.op_type.clone());
            used_numerics.extend(match_consumer::<F>(op_type).used_numerics().iter())
        }

        let field_tensor_map = graph
            .tensor_map
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    Array::from_shape_vec(
                        v.shape(),
                        v.iter().map(|x| to_field::<F>(x.clone())).collect(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        (used_numerics, field_tensor_map)
    }

    fn construct(graph: Graph) -> Self;

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
                            .collect::<Result<Vec<_>, ErrorFront>>()?;
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
                    .collect::<Result<HashMap<_, _>, ErrorFront>>()?;
                Ok(assigned_tensors)
            },
        )?)
    }

    fn assign_constants(
        &self,
        mut layouter: impl Layouter<F>,
        config: Rc<NumericConfig>,
    ) -> Result<HashMap<Int, CellRc<F>>, Error> {
        let sf = config.scale_factor;
        // let min_val = config.min_val;
        // let max_val = config.max_val;

        Ok(layouter.assign_region(
            || "constants",
            |mut region| {
                let mut constants: HashMap<Int, CellRc<F>> = HashMap::new();

                let vals = vec![0 as Int, 1, sf as Int /*min_val, max_val*/];
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
        )?)
    }

    fn run(&self) -> Result<Tensor, ShapeError>;
}
