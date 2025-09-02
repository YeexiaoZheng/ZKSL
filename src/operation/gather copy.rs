use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
};
use ndarray::{s, Array, IxDyn, ShapeError};

use crate::{
    numeric::{
        dot_vec::DotVecLayouter, nonlinear::gather::GatherLookUp, update::UpdateLayouter,
        NumericConfig, NumericConsumer, NumericLayout, NumericType,
    },
    utils::{
        helpers::{to_primitive, AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{fdiv, Int},
    },
};

use super::Operation;

#[derive(Clone, Debug, Default)]
pub struct GatherChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> GatherChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    // This function is used for non-circuit forward
    // Lookup rows in the weight tensor corresponding to the indices in the input tensor
    pub fn forward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let input = &inputs[0];
        let weight = &inputs[1];
        let weight_shape = (weight.shape()[0], weight.shape()[1]);
        for &idx in input.iter() {
            // println!("idx:{}, input:{}, weight 0:{}", idx, input, weight_shape.0);
            assert!(idx <= weight_shape.0 as Int);
        }

        let outputs = input
            .iter()
            .map(|&idx| weight.slice(s![idx as usize, ..]).to_owned())
            .flatten()
            .collect::<Vec<_>>();

        // println!("output{:?}", &outputs);

        Ok(vec![Array::from_shape_vec(
            IxDyn(&[input.shape()[0], weight.shape()[1]]),
            outputs,
        )
        .unwrap()])
    }

    // This function is used for non-circuit backward
    // Here, the gradient is propagated instead of the updated weight
    // Therefore, the non-circuit backward operation does not need to perform any updates.
    // It only needs to return the gradient.
    pub fn backward(
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let inpgrad = &inputs[0];
        let input = &inputs[1];
        let weight = &inputs[2];
        for &idx in input.iter() {
            // println!("idx:{}, input:{}, weight 0:{}", idx, input, weight_shape.0);
            assert!(idx <= weight.shape()[0] as Int);
        }
        let mut new_weight = weight.clone();
        let input = input.into_iter().collect::<Vec<_>>();
        for (i, inpgrad) in inpgrad.outer_iter().enumerate() {
            let emb_idx = input[i].clone();
            let weight_slice = new_weight.slice(s![emb_idx as usize, ..]).to_owned();
            let new_weight_slice_vec =
                weight_slice - inpgrad.mapv(|x| fdiv(x, numeric_config.reciprocal_learning_rate));
            for (j, x) in new_weight_slice_vec.into_iter().enumerate() {
                new_weight[[emb_idx as usize, j]] = x;
            }
        }

        Ok(vec![inpgrad.clone(), new_weight])
    }
}

impl<F: PrimeField> Operation<F> for GatherChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        _constants: &BTreeMap<Int, CellRc<F>>,
        random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // Check input shape
        let input = inputs[0].clone();
        let weight = inputs[1].clone();
        let weight_shape = weight.shape();

        let emb_dim = weight_shape[1];
        let random_vector = random[0..emb_dim]
            .iter()
            .map(|x| x.as_ref())
            .collect::<Vec<_>>();

        let dot = DotVecLayouter::construct(self.numeric_config.clone());
        let gather = GatherLookUp::construct(self.numeric_config.clone());

        // Because the input is a tensor of indices, the primitive values can't be visible in keygen
        // So we need to use other methods to implement embedding
        // We use copy primitive values to the output tensor, and then assign them to the corresponding columns
        // But its a little bit tricky and hacker
        let primitive_input = input
            .iter()
            .map(|x| {
                let mut idx = -1;
                x.value().map(|x| idx = to_primitive(x));
                idx
            })
            .collect::<Vec<_>>();

        let mut _outputs = vec![];
        if primitive_input[0] == -1 {
            _outputs = vec![Value::unknown(); input.shape()[0] * weight_shape[1]];
        } else {
            // Check input
            for &idx in primitive_input.iter() {
                assert!(idx <= weight_shape[0] as Int);
            }
            // Extract all indices from input=
            for idx in primitive_input.iter() {
                _outputs.extend(
                    weight
                        .slice(s![*idx as usize, ..])
                        .iter()
                        .map(|x| x.value().map(|&x| x)),
                );
            }
        }

        let outputs = layouter
            .assign_region(
                || "assign new gather",
                |mut region| {
                    let columns = self.numeric_config.columns.clone();
                    Ok(_outputs
                        .clone()
                        .into_iter()
                        .enumerate()
                        .map(|(i, x)| {
                            let column = columns[i % columns.len()];
                            let offset = i / columns.len();
                            region
                                .assign_advice(|| "fill new region", column, offset, || x)
                                .unwrap()
                        })
                        .collect::<Vec<_>>())
                },
            )
            .unwrap();

        let outputs = Array::from_shape_vec(
            IxDyn(&[input.shape()[0], weight.shape()[1]]),
            outputs.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>(),
        )?;

        // Constrain the embedding lookup
        layouter
            .assign_region(
                || "constrain embedding lookup",
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;
                    let mut rand_opts = vec![];
                    for opt in outputs.outer_iter() {
                        let (rand_opt, offset) = dot
                            .layout(
                                region,
                                row_offset,
                                &vec![
                                    opt.iter().map(|x| x.as_ref()).collect(),
                                    random_vector.clone(),
                                ],
                                &vec![],
                            )
                            .unwrap();
                        rand_opts.push(rand_opt[0].clone());
                        row_offset = offset;
                    }
                    gather
                        .layout(
                            region,
                            row_offset,
                            &vec![
                                input.iter().map(|x| x.as_ref()).collect(),
                                rand_opts.iter().collect(),
                            ],
                            &vec![],
                        )
                        .unwrap();
                    Ok(())
                },
            )
            .unwrap();

        Ok(vec![outputs])
    }

    fn backward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // Check input shape
        // assert_eq!(inputs.len(), 3);
        let inpgrad = inputs[0].clone();
        let input = inputs[1].clone();
        let weight = inputs[2].clone();
        // let input_shape = input.shape();

        let emb_dim = weight.shape()[1];
        let random_vector = random[0..emb_dim]
            .iter()
            .map(|x| x.as_ref())
            .collect::<Vec<_>>();

        // let dot = DotVecLayouter::construct(self.numeric_config.clone());
        // let gather = GatherLookUp::construct(self.numeric_config.clone());

        // Because the input is a tensor of indices, the primitive values can't be visible in keygen
        // So we need to use other methods to implement embedding
        // We use copy primitive values to the output tensor, and then assign them to the corresponding columns
        // But its a little bit tricky and hacker
        let primitive_input = input
            .iter()
            .map(|x| {
                let mut idx = -1;
                x.value().map(|x| idx = to_primitive(x));
                idx
            })
            .collect::<Vec<_>>();

        let weight_shape = weight.shape();
        let inpgrad_shape = inpgrad.shape();
        assert_eq!(weight_shape[1], inpgrad_shape[1]);
        let input = input
            .into_iter()
            .map(|x| x.value().map(|x| to_primitive(x)))
            .collect::<Vec<_>>();
        for v in input.iter() {
            v.map(|x| {
                assert!(x <= weight_shape[0] as Int);
            });
        }

        // Get constants
        let constants = self.get_default_constants(constants);
        let zero = constants[0];
        let zero = zero.clone();

        // Initialize numeric layouters
        let update = UpdateLayouter::construct(self.numeric_config.clone());

        // Weight backward pass
        let new_weight = layouter
            .assign_region(
                || "gather",
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;

                    let mut new_weight = weight.mapv(|x| x.as_ref().clone());

                    if primitive_input[0] != -1 {
                        for (idx, inpgrad) in inpgrad.outer_iter().enumerate() {
                            let grad_vec = inpgrad.clone().into_iter().collect::<Vec<_>>();
                            input[idx].map(|i| {
                                let weight_slice = new_weight.slice(s![i as usize, ..]);
                                let (new_weight_slice_vec, tmp_offset) = match update.layout(
                                    region,
                                    row_offset,
                                    &vec![
                                        weight_slice.iter().collect::<Vec<_>>(),
                                        grad_vec.iter().map(|x| x.as_ref()).collect::<Vec<_>>(),
                                    ],
                                    &constants,
                                ) {
                                    Ok(grad) => grad,
                                    Err(_) => {
                                        println!("Error in add layout");
                                        (vec![zero.clone(); weight_slice.len()], row_offset)
                                    }
                                };
                                row_offset = tmp_offset;
                                for (j, x) in new_weight_slice_vec.into_iter().enumerate() {
                                    new_weight[[i as usize, j]] = x;
                                }
                            });
                        }
                    }
                    Ok(new_weight)
                },
            )
            .unwrap();

        Ok(vec![inpgrad.to_owned(), new_weight.mapv(|x| Rc::new(x))])
    }
}

impl<F: PrimeField> NumericConsumer for GatherChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![NumericType::Add, NumericType::Gather]
    }
}
