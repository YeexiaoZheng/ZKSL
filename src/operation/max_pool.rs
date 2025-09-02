// TODO: Speed up Depthwise operations with Freivald's algorithm
use std::{collections::BTreeMap, marker::PhantomData, rc::Rc, vec};

use super::Operation;
use crate::operation::conv::splat_input;
use crate::utils::helpers::to_primitive;
use crate::{
    numeric::{max::MaxLayouter, NumericConfig, NumericConsumer, NumericLayout, NumericType},
    stage::assign::Assign,
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::Int,
    },
};
use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{Array, ShapeError};

pub struct MaxPoolChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

impl<F: PrimeField> MaxPoolChip<F> {
    pub fn construct(numeric_config: Rc<NumericConfig>) -> Self {
        Self {
            numeric_config,
            _marker: PhantomData,
        }
    }

    // This function is used for non-circuit forward
    pub fn forward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let stride = match attributes.get("strides") {
            Some(x) => (x[0] as usize, x[1] as usize),
            None => panic!("attributes not found!"),
        };
        let (ph, pw) = match attributes.get("pads") {
            Some(x) => (
                (x[0] as usize, x[1] as usize),
                (x[2] as usize, x[3] as usize),
            ),
            None => panic!("attributes not found!"),
        };
        let kernal_shape = match attributes.get("kernel_shape") {
            Some(x) => (x[0] as usize, x[1] as usize),
            None => panic!("attributes not found!"),
        };

        // input: batch * C_in * H * W
        let input = &inputs[0];

        let (batch_size, c_in, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (hh, ww) = kernal_shape;
        let (si, sj) = stride;
        let (oh, ow) = (
            (h + ph.0 + ph.1 - hh) / si + 1,
            (w + pw.0 + pw.1 - ww) / sj + 1,
        );

        let splat_inp = splat_input(&input, stride, hh, ww, oh, ow);

        let output = splat_inp
            .iter()
            .map(|v| *v.iter().max().unwrap())
            .collect::<Vec<_>>();

        let out_shape = vec![batch_size, c_in, oh, ow];
        let output = Array::from_shape_vec(out_shape, output)?.to_owned();
        // println!("max_pool non-circuit forward output: {:?}", output.shape());

        Ok(vec![output])
    }

    // This function is used for non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        _numeric_config: &NumericConfig,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Tensor>, ShapeError> {
        let stride = match attributes.get("strides") {
            Some(x) => (x[0] as usize, x[1] as usize),
            None => panic!("attributes not found!"),
        };
        let (_ph, _pw) = match attributes.get("pads") {
            Some(x) => (
                (x[0] as usize, x[1] as usize),
                (x[2] as usize, x[3] as usize),
            ),
            None => panic!("attributes not found!"),
        };
        let kernal_shape = match attributes.get("kernel_shape") {
            Some(x) => (x[0] as usize, x[1] as usize),
            None => panic!("attributes not found!"),
        };

        // batch * C_in * H * W
        let input = &inputs[1].clone();
        // batch * C_out * OH * OW
        let inpgrad = &inputs[0].clone();

        let (batch_size, c_in, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (hh, ww) = kernal_shape;
        let (si, sj) = stride;
        let (oh, ow) = (inpgrad.shape()[2], inpgrad.shape()[3]);

        let mut grad_input = Array::zeros(input.raw_dim());

        for b in 0..batch_size {
            for c in 0..c_in {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut max_val = Int::MIN;
                        let mut max_pos = (0, 0);
                        for x in 0..hh {
                            for y in 0..ww {
                                let ix = i * si + x;
                                let jy = j * sj + y;
                                if ix < h && jy < w {
                                    if input[[b, c, ix, jy]] > max_val {
                                        max_val = input[[b, c, ix, jy]];
                                        max_pos = (ix, jy);
                                    }
                                }
                            }
                        }
                        grad_input[[b, c, max_pos.0, max_pos.1]] = inpgrad[[b, c, i, j]];
                    }
                }
            }
        }
        // println!("max_pool non-circuit backward output: {:?}", grad_input.shape());

        Ok(vec![grad_input])
    }
}

impl<F: PrimeField> Assign<F> for MaxPoolChip<F> {}
impl<F: PrimeField> Operation<F> for MaxPoolChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let stride = match attributes.get("strides") {
            Some(x) => (x[0] as usize, x[1] as usize),
            None => panic!("attributes not found!"),
        };
        let (ph, pw) = match attributes.get("pads") {
            Some(x) => (
                (x[0] as usize, x[1] as usize),
                (x[2] as usize, x[3] as usize),
            ),
            None => panic!("attributes not found!"),
        };
        let kernal_shape = match attributes.get("kernel_shape") {
            Some(x) => (x[0] as usize, x[1] as usize),
            None => panic!("attributes not found!"),
        };

        let zero = constants.get(&0).unwrap();
        // input: batch * C_in * H * W
        let input = &inputs[0];

        let (batch_size, c_in, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (hh, ww) = kernal_shape;
        let (si, sj) = stride;
        let (oh, ow) = (
            (h + ph.0 + ph.1 - hh) / si + 1,
            (w + pw.0 + pw.1 - ww) / sj + 1,
        );

        let input = inputs
            .iter()
            .into_iter()
            .map(|x| x.map(|x| Rc::clone(x)))
            .collect::<Vec<_>>();

        let splat_inp = splat_input(&input[0], stride, hh, ww, oh, ow);

        let max_chip = MaxLayouter::construct(self.numeric_config.clone());
        let output = layouter
            .assign_region(
                || format!("max pool forward"),
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;
                    let mut output = vec![];
                    splat_inp.iter().for_each(|v| {
                        let inpt = v.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                        let outp = max_chip
                            .layout(region, row_offset, &vec![inpt], &vec![zero.as_ref()])
                            .unwrap();
                        output.extend(outp.0);
                        row_offset = outp.1;
                    });
                    Ok(output)
                },
            )
            .unwrap();

        let out_shape = vec![batch_size, c_in, oh, ow];
        let output = output.iter().map(|x| Rc::new(x.clone())).collect();
        let output = Array::from_shape_vec(ndarray::IxDyn(&out_shape), output).unwrap();
        // println!("max pool circuit forward output: {:?}", output.shape());
        Ok(vec![output])
    }

    fn backward(
        &self,
        mut _layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        _random: &Vec<CellRc<F>>,
        attributes: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<AssignedTensor<F>>, ShapeError> {
        let stride = match attributes.get("strides") {
            Some(x) => (x[0] as usize, x[1] as usize),
            None => panic!("attributes not found!"),
        };
        let kernal_shape = match attributes.get("kernel_shape") {
            Some(x) => (x[0] as usize, x[1] as usize),
            None => panic!("attributes not found!"),
        };
        let zero = constants.get(&0).unwrap();

        let inpgrad = inputs[0].clone();
        let input = inputs[1].clone();

        let (batch_size, channel, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (oh, ow) = (inpgrad.shape()[2], inpgrad.shape()[3]);
        let (hh, ww) = kernal_shape;
        let (si, sj) = stride;

        let mut grad_input = Array::from_elem(input.raw_dim(), zero.clone());

        for b in 0..batch_size {
            for c in 0..channel {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut max_val = Int::MIN;
                        let mut max_pos = (0, 0);
                        for x in 0..hh {
                            for y in 0..ww {
                                let ix = i * si + x;
                                let jy = j * sj + y;
                                if ix < h && jy < w {
                                    let cur = &input[[b, c, ix, jy]];
                                    let mut val = -1;
                                    cur.value().map(|x| val = to_primitive(x));
                                    if val > max_val {
                                        max_val = val;
                                        max_pos = (ix, jy);
                                    }
                                }
                            }
                        }
                        grad_input[[b, c, max_pos.0, max_pos.1]] = inpgrad[[b, c, i, j]].clone();
                    }
                }
            }
        }
        // println!("max pool circuit backward output: {:?}", grad_input.shape());

        Ok(vec![grad_input])
    }
}

impl<F: PrimeField> NumericConsumer for MaxPoolChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::Sum,
            NumericType::Add,
            NumericType::Update,
            NumericType::DotVec,
            NumericType::DivSF,
            NumericType::Max,
        ]
    }
}
