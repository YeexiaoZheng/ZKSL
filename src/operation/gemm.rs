use std::{collections::BTreeMap, marker::PhantomData, rc::Rc, vec};

use halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::ff::PrimeField,
};
use ndarray::{s, Array, Axis, IxDyn, ShapeError};

use crate::{
    numeric::{
        add::AddLayouter, div_sf::DivSFLayouter, dot_vec::DotVecLayouter, update::UpdateLayouter,
        NumericConfig, NumericConsumer, NumericLayout, NumericType,
    },
    stage::assign::Assign,
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor, ValueTensor},
        math::{fdiv, Int},
    },
};

use super::Operation;

#[derive(Clone, Debug, Default)]
pub struct GemmChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> GemmChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    // This function is used for non-circuit forward
    pub fn forward(
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let input = &inputs[0];
        let weight = &inputs[1].clone();
        let bias = &inputs[2].flatten().clone();
        let input_shape = (input.shape()[0], input.shape()[1]);
        let weight_shape = (weight.shape()[0], weight.shape()[1]);
        // println!("gemm input shape: {:?}", input_shape);
        // println!("gemm weight shape: {:?}", weight_shape);
        assert_eq!(input_shape.1, weight_shape.0);

        let input = input.to_shape(input_shape)?;
        let weight = weight.to_shape(weight_shape)?;
        let output = input.dot(&weight);
        let output = output.mapv(|x| fdiv(x, numeric_config.scale_factor as Int));
        let output = output + bias;

        Ok(vec![output.into_dyn()])
    }

    // This function is used for non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let inpgrad = &inputs[0].clone();
        let input = &inputs[1].clone();
        let weight = &inputs[2].clone();
        let inpgrad_shape = (inpgrad.shape()[0], inpgrad.shape()[1]);
        let input_shape = (input.shape()[0], input.shape()[1]);
        let weight_shape = (weight.shape()[0], weight.shape()[1]);
        assert_eq!(input_shape.1, weight_shape.0);
        assert_eq!(inpgrad_shape.1, weight_shape.1);

        let inpgrad = inpgrad.to_shape(inpgrad_shape)?;
        let input = input.to_shape(input_shape)?;
        let weight = weight.to_shape(weight_shape)?;

        let out_grad = inpgrad
            .dot(&weight.t())
            .into_dyn()
            .mapv(|x| fdiv(x, numeric_config.scale_factor as Int));

        let weight_grad = input
            .t()
            .dot(&inpgrad)
            .into_dyn()
            .mapv(|x| fdiv(x, numeric_config.scale_factor as Int));

        // let bias_grad = inpgrad.sum_axis(Axis(0)).into_dyn();

        let new_weight =
            weight - weight_grad.mapv(|x| fdiv(x, numeric_config.reciprocal_learning_rate));

        // let new_bias = bias - bias_grad.mapv(|x| fdiv(x, numeric_config.reciprocal_learning_rate));

        Ok(vec![
            out_grad,
            new_weight.to_owned(),
            // new_bias.to_owned(),
        ])
    }
}

fn dot_matrix<F: PrimeField>(
    input: AssignedTensorRef<F>,
    weight: AssignedTensorRef<F>,
) -> ValueTensor<F> {
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    assert_eq!(input_shape.len(), 2);
    assert_eq!(input_shape.len(), weight_shape.len());
    assert_eq!(input_shape[1], weight_shape[0]);

    let mut output = vec![];
    for i in 0..input_shape[0] {
        for j in 0..weight_shape[1] {
            let input = input.slice(s![i, ..]).to_vec();
            let weight = weight.slice(s![.., j]).to_vec();
            let dot = input
                .iter()
                .zip(weight.iter())
                .map(|(x, y)| x.value().copied() * y.value().copied())
                .fold(Value::known(F::ZERO), |acc, x| acc + x);
            output.push(dot);
        }
    }

    Array::from_shape_vec(IxDyn(&[input_shape[0], weight_shape[1]]), output).unwrap()
}

impl<F: PrimeField> Assign<F> for GemmChip<F> {}

impl<F: PrimeField> Operation<F> for GemmChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        random: &Vec<CellRc<F>>,
        _attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        // Check input shape
        let input = inputs[0].clone();
        let weight = inputs[1].clone();
        let bias = inputs[2].clone();
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        assert_eq!(input_shape.len(), 2);
        assert_eq!(input_shape.len(), weight_shape.len());
        assert_eq!(input_shape[1], weight_shape[0]);

        // Get constants
        // let constants = self.get_default_constants(constants);
        let constants = self.get_constants(
            constants,
            vec![0, 1, self.numeric_config.scale_factor as Int],
        );

        // Initialize numeric layouters
        let dot = DotVecLayouter::construct(self.numeric_config.clone());
        let div_sf = DivSFLayouter::construct(self.numeric_config.clone());
        let add = AddLayouter::construct(self.numeric_config.clone());

        /*
                let dot_res = layouter
                    .assign_region(
                        || "GEMM normal dot",
                        |mut region| {
                            let region = &mut region;
                            // Forward pass
                            let mut row_offset = 0;
                            let mut outputs = vec![];
                            for i in 0..input_shape[0] {
                                for j in 0..weight_shape[1] {
                                    let input = input
                                        .slice(s![i, ..])
                                        .into_iter()
                                        .map(|x| x.as_ref())
                                        .collect::<Vec<_>>();
                                    let weight = weight
                                        .slice(s![.., j])
                                        .into_iter()
                                        .map(|x| x.as_ref())
                                        .collect::<Vec<_>>();
                                    let output = match dot.layout(
                                        region,
                                        row_offset,
                                        &vec![input, weight],
                                        &constants,
                                    ) {
                                        Ok(output) => output,
                                        Err(e) => panic!("Error in GemmChip.dot: {:?}", e),
                                    };
                                    row_offset = output.1;
                                    outputs.extend(output.0);
                                }
                            }
                            Ok(outputs)
                        },
                    )
                    .unwrap();
        */


        let dot_res: Vec<_> = {
            // Compute the result
            let dot_res = dot_matrix(input.clone(), weight.clone());
            // Assign th result
            let dot_res = self
                .assign_value_tensor(
                    layouter.namespace(|| "assign gemm result"),
                    &self.numeric_config.columns,
                    &dot_res,
                )
                .unwrap();
            let dot_res = dot_res.mapv(|x| x.as_ref().clone());
            // Get random vector
            let r = random[0..dot_res.shape()[1]].to_vec();
            let r = r.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
            // Check dot res
            layouter
                .assign_region(
                    || format!("GEMM random dot"),
                    |mut region| {
                        let region = &mut region;
                        // Forward pass
                        let mut row_offset = 0;
                        // Check res * r
                        let mut res_r = vec![];
                        for i in 0..dot_res.shape()[0] {
                            let res_row = dot_res.index_axis(Axis(0), i);
                            let res_row = res_row.into_iter().collect::<Vec<_>>();
                            let output = dot
                                .layout(region, row_offset, &vec![res_row, r.clone()], &constants)
                                .unwrap();
                            row_offset = output.1;
                            res_r.extend(output.0);
                        }

                        // Check input * weight * r
                        let mut weight_r = vec![];
                        for i in 0..weight_shape[0] {
                            let weight_row = weight.slice(s![i, ..]).to_vec();
                            let weight_row =
                                weight_row.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                            let output = dot
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![weight_row, r.clone()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = output.1;
                            weight_r.extend(output.0);
                        }

                        let mut input_weight_r = vec![];
                        for i in 0..input_shape[0] {
                            let input_row = input.index_axis(Axis(0), i);
                            let input_row = input_row.iter().map(|x| x.as_ref()).collect();
                            let output = dot
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![input_row, weight_r.clone().iter().collect()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = output.1;
                            input_weight_r.extend(output.0);
                        }
                        // Constrain equal res * r = input * weight * r
                        for (i, j) in input_weight_r.iter().zip(res_r.iter()) {
                            region.constrain_equal(i.cell(), j.cell()).unwrap();
                        }

                        // END DOT PRODUCT
                        Ok(())
                    },
                )
                .unwrap();
            dot_res.into_iter().collect()
        };

        // Divide by scale factor & Add bias
        let outputs = layouter
            .assign_region(
                || format!("GEMM forward{}", weight.len()),
                |mut region| {
                    let region = &mut region;
                    // Forward pass
                    let mut row_offset = 0;

                    // Divide by scale factor
                    let outputs = match div_sf.layout(
                        region,
                        row_offset,
                        &vec![dot_res.iter().collect()],
                        &constants,
                    ) {
                        Ok(output) => output,
                        Err(e) => panic!("Error in GemmChip.div_sf: {:?}", e),
                    };
                    row_offset = outputs.1;
                    let outputs = outputs.0;

                    // Add bias
                    let outputs = outputs
                        .chunks(bias.len())
                        .map(|chunk| {
                            let output = match add.layout(
                                region,
                                row_offset,
                                &vec![
                                    chunk.iter().collect(),
                                    bias.iter().map(|x| x.as_ref()).collect(),
                                ],
                                &constants,
                            ) {
                                Ok(output) => output,
                                Err(e) => panic!("Error in GemmChip.add: {:?}", e),
                            };
                            row_offset = output.1;
                            output.0
                        })
                        .flatten()
                        .collect::<Vec<_>>();

                    Ok(outputs)
                },
            )
            .unwrap();

        Ok(vec![Array::from_shape_vec(
            IxDyn(&[input_shape[0], weight_shape[1]]),
            outputs.into_iter().map(|x| Rc::new(x)).collect(),
        )?])
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
        let input_t = inputs[1].t().clone();
        let weight = inputs[2].clone();
        let weight_t = inputs[2].t().clone();
        let inpgrad_shape = inpgrad.shape();
        let input_t_shape = input_t.shape();
        let weight_t_shape = weight_t.shape();
        assert_eq!(inpgrad_shape.len(), weight_t_shape.len());
        assert_eq!(input_t_shape[1], inpgrad_shape[0]);
        assert_eq!(inpgrad_shape[1], weight_t_shape[0]);

        // Get constants
        let zero = constants.get(&0).unwrap().clone();
        let one = constants.get(&1).unwrap().clone();
        let sf = constants
            .get(&(self.numeric_config.scale_factor as Int))
            .unwrap()
            .clone();
        let constants = vec![zero.as_ref(), one.as_ref(), sf.as_ref()];

        // Initialize numeric chip
        let dot = DotVecLayouter::construct(self.numeric_config.clone());
        let div_sf = DivSFLayouter::construct(self.numeric_config.clone());
        let update = UpdateLayouter::construct(self.numeric_config.clone());

        // input backward pass
        let outgrad: Vec<_> = {
            // Compute the result
            let dot_res = dot_matrix(inpgrad.clone(), weight_t.clone());
            // Assign th result
            let dot_res = self
                .assign_value_tensor(
                    layouter.namespace(|| "assign inpgrad @ weight_t result"),
                    &self.numeric_config.columns,
                    &dot_res,
                )
                .unwrap();
            let dot_res = dot_res.mapv(|x| x.as_ref().clone());
            // Get random vector
            let r = random[0..dot_res.shape()[1]].to_vec();
            let r = r.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
            // Check dot res
            layouter
                .assign_region(
                    || format!("GEMM inpgrad @ weight_t random dot"),
                    |mut region| {
                        let region = &mut region;
                        // Forward pass
                        let mut row_offset = 0;
                        // Check res * r
                        let mut res_r = vec![];
                        for i in 0..dot_res.shape()[0] {
                            let res_row = dot_res.index_axis(Axis(0), i);
                            let res_row = res_row.into_iter().collect::<Vec<_>>();
                            let output = dot
                                .layout(region, row_offset, &vec![res_row, r.clone()], &constants)
                                .unwrap();
                            row_offset = output.1;
                            res_r.extend(output.0);
                        }

                        // Check inpgrad * weight.T * r
                        let mut weight_t_r = vec![];
                        for i in 0..weight_t_shape[0] {
                            let weight_row = weight_t.slice(s![i, ..]).to_vec();
                            let weight_row =
                                weight_row.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                            let output = dot
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![weight_row, r.clone()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = output.1;
                            weight_t_r.extend(output.0);
                        }

                        let mut inpgrad_weight_t_r = vec![];
                        for i in 0..inpgrad_shape[0] {
                            let input_row = inpgrad.index_axis(Axis(0), i);
                            let input_row = input_row.iter().map(|x| x.as_ref()).collect();
                            let output = dot
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![input_row, weight_t_r.clone().iter().collect()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = output.1;
                            inpgrad_weight_t_r.extend(output.0);
                        }
                        // Constrain equal res * r = inpgrad * weight.T * r
                        for (i, j) in inpgrad_weight_t_r.iter().zip(res_r.iter()) {
                            region.constrain_equal(i.cell(), j.cell()).unwrap();
                        }

                        // END DOT PRODUCT
                        Ok(())
                    },
                )
                .unwrap();
            dot_res.into_iter().collect()
        };

        // weight backward pass
        let weight_grad: Vec<_> = {
            // Compute the result
            let dot_res = dot_matrix(input_t.clone(), inpgrad.clone());
            // Assign th result
            let dot_res = self
                .assign_value_tensor(
                    layouter.namespace(|| "assign input_t @ inpgrad result"),
                    &self.numeric_config.columns,
                    &dot_res,
                )
                .unwrap();
            let dot_res = dot_res.mapv(|x| x.as_ref().clone());
            // Get random vector
            let r = random[0..dot_res.shape()[1]].to_vec();
            let r = r.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
            // Check dot res
            layouter
                .assign_region(
                    || format!("GEMM input_t @ inpgrad random dot"),
                    |mut region| {
                        let region = &mut region;
                        // Forward pass
                        let mut row_offset = 0;
                        // Check res * r
                        let mut res_r = vec![];
                        for i in 0..dot_res.shape()[0] {
                            let res_row = dot_res.index_axis(Axis(0), i);
                            let res_row = res_row.into_iter().collect::<Vec<_>>();
                            let output = dot
                                .layout(region, row_offset, &vec![res_row, r.clone()], &constants)
                                .unwrap();
                            row_offset = output.1;
                            res_r.extend(output.0);
                        }

                        // Check input.T * inpgrad * r
                        let mut inpgrad_r = vec![];
                        for i in 0..inpgrad_shape[0] {
                            let inpgrad_row = inpgrad.slice(s![i, ..]).to_vec();
                            let inpgrad_row =
                                inpgrad_row.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                            let output = dot
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![inpgrad_row, r.clone()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = output.1;
                            inpgrad_r.extend(output.0);
                        }

                        let mut input_t_inpgrad_r = vec![];
                        for i in 0..input_t_shape[0] {
                            let input_t_row = input_t.index_axis(Axis(0), i);
                            let input_t_row = input_t_row.iter().map(|x| x.as_ref()).collect();
                            let output = dot
                                .layout(
                                    region,
                                    row_offset,
                                    &vec![input_t_row, inpgrad_r.clone().iter().collect()],
                                    &constants,
                                )
                                .unwrap();
                            row_offset = output.1;
                            input_t_inpgrad_r.extend(output.0);
                        }
                        // Constrain equal res * r = input.T * inpgrad * r
                        for (i, j) in input_t_inpgrad_r.iter().zip(res_r.iter()) {
                            region.constrain_equal(i.cell(), j.cell()).unwrap();
                        }

                        // END DOT PRODUCT
                        Ok(())
                    },
                )
                .unwrap();
            dot_res.into_iter().collect()
        };

        let (outgrad, new_weight) = layouter
            .assign_region(
                || format!("GEMM forward{}", weight.len()),
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;

                    // Divide by scale factor
                    let outgrad = match div_sf.layout(
                        region,
                        row_offset,
                        &vec![outgrad.iter().collect()],
                        &constants,
                    ) {
                        Ok(output) => output,
                        Err(e) => panic!("Error in GemmChip.div_sf: {:?}", e),
                    };
                    row_offset = outgrad.1;
                    let outgrad = outgrad.0;

                    // Divide by scale factor
                    let weight_grad = match div_sf.layout(
                        region,
                        row_offset,
                        &vec![weight_grad.iter().collect()],
                        &constants,
                    ) {
                        Ok(output) => output,
                        Err(e) => panic!("Error in GemmChip.div_sf: {:?}", e),
                    };
                    row_offset = weight_grad.1;
                    let weight_grad = weight_grad.0;

                    // Update weight
                    let new_weight = match update.layout(
                        region,
                        row_offset,
                        &vec![
                            weight.iter().map(|x| x.as_ref()).collect(),
                            weight_grad.iter().collect(),
                        ],
                        &constants,
                    ) {
                        Ok(output) => output,
                        Err(e) => panic!("Error in GemmChip.update: {:?}", e),
                    };
                    let new_weight = new_weight.0;

                    Ok((outgrad, new_weight))
                },
            )
            .unwrap();

        Ok(vec![
            Array::from_shape_vec(
                IxDyn(&[inpgrad_shape[0], weight_t_shape[1]]),
                outgrad.into_iter().map(|x| Rc::new(x)).collect(),
            )?,
            Array::from_shape_vec(
                IxDyn(&[input_t_shape[0], inpgrad_shape[1]]),
                new_weight.into_iter().map(|x| Rc::new(x)).collect(),
            )?,
        ])
    }
}

impl<F: PrimeField> NumericConsumer for GemmChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::DotVec,
            NumericType::Sum,
            NumericType::Add,
            NumericType::DivSF,
            NumericType::NaturalLookUp,
        ]
    }
}
