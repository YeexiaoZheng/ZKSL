// TODO: Speed up Depthwise operations with Freivald's algorithm
use std::{collections::BTreeMap, marker::PhantomData, rc::Rc, vec};

use super::Operation;
use crate::numeric::sum::SumLayouter;
use crate::numeric::update::UpdateLayouter;
use crate::operation::gemm::GemmChip;
use crate::utils::helpers::to_field;
use crate::{
    numeric::{add::AddLayouter, NumericConfig, NumericConsumer, NumericLayout, NumericType},
    stage::assign::Assign,
    utils::{
        helpers::{AssignedTensor, AssignedTensorRef, CellRc, Tensor},
        math::{fdiv, Int},
    },
};
use halo2_proofs::circuit::AssignedCell;
use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField};
use ndarray::{s, Array, Array4, Axis, IxDyn, ShapeError, Slice};

#[derive(Default, Clone, Copy, Eq, PartialEq, Debug)]
pub enum PaddingEnum {
    #[default]
    Same,
    Valid,
}

pub struct ConvChip<F: PrimeField> {
    pub numeric_config: Rc<NumericConfig>,
    pub _marker: PhantomData<F>,
}

pub fn pad<G: Clone>(
    input: &Array<G, IxDyn>,
    padding: Vec<[usize; 2]>,
    pad_val: &G,
) -> Array<G, IxDyn> {
    let tmp = input.iter().collect();
    let input = Array::from_shape_vec(input.raw_dim(), tmp).unwrap();
    assert_eq!(input.ndim(), padding.len());
    let mut padded_shape = input.raw_dim();
    for (ax, (&ax_len, &[pad_lo, pad_hi])) in input.shape().iter().zip(&padding).enumerate() {
        padded_shape[ax] = ax_len + pad_lo + pad_hi;
    }

    let mut padded = Array::from_elem(padded_shape, pad_val);
    let padded_dim = padded.raw_dim();
    {
        // Select portion of padded array that needs to be copied from the
        // original array.
        let mut orig_portion = padded.view_mut();
        for (ax, &[pad_lo, pad_hi]) in padding.iter().enumerate() {
            orig_portion.slice_axis_inplace(
                Axis(ax),
                Slice::from(pad_lo as isize..padded_dim[ax] as isize - (pad_hi as isize)),
            );
        }
        // Copy the data from the original array.
        orig_portion.assign(&input.view());
    }

    let dim = padded.raw_dim();
    let tmp = padded.into_iter().map(|x| x.clone()).collect();
    let padded = Array::from_shape_vec(dim, tmp).unwrap();

    padded
}

pub fn splat<G: Clone + std::fmt::Debug>(
    inputs: &Vec<Array<G, IxDyn>>,
    zero: G,
    stride: (usize, usize),
    ph: (usize, usize),
    pw: (usize, usize),
) -> (Vec<Vec<G>>, Vec<Vec<G>>, Vec<G>) {
    assert!(inputs.len() <= 3);

    let input = &inputs[0];
    let weight = &inputs[1];
    let biases = if inputs.len() == 3 {
        &inputs[2]
    } else {
        &Array::from_elem(IxDyn(&vec![1]), zero.clone())
    };

    let (batch_size, c_in, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (c_out, hh, ww) = (weight.shape()[0], weight.shape()[2], weight.shape()[3]);
    let (si, sj) = stride;
    let (oh, ow) = (
        (h + ph.0 + ph.1 - hh) / si + 1,
        (w + pw.0 + pw.1 - ww) / sj + 1,
    );

    // B, C, H, W
    assert_eq!(input.shape().len(), 4);

    let paddings = vec![[0, 0], [0, 0], [ph.0, ph.1], [pw.0, pw.1]];
    let inp_pad = pad(&input, paddings, &zero);

    let mut input_cells = vec![];
    let mut weights_cells = vec![];
    let mut biases_cells = vec![];
    let mut input_row_idx = 0;
    let mut weight_row_idx = 0;

    // (C_out, C_in * HH * WW)
    for cout in 0..c_out {
        weights_cells.push(vec![]);
        for ci in 0..hh {
            for cj in 0..ww {
                for ck in 0..c_in {
                    weights_cells[weight_row_idx].push(weight[[cout, ck, ci, cj]].clone());
                }
            }
        }
        weight_row_idx += 1;
    }

    // (oh * ow, C_in * HH * WW)
    for batch in 0..batch_size {
        for i in 0..oh {
            for j in 0..ow {
                input_cells.push(vec![]);
                for ci in 0..hh {
                    for cj in 0..ww {
                        for ck in 0..c_in {
                            let idx_i = i * si + ci;
                            let idx_j = j * sj + cj;
                            input_cells[input_row_idx]
                                .push(inp_pad[[batch, ck, idx_i, idx_j]].clone());
                        }
                    }
                }
                input_row_idx += 1;
            }
        }
    }

    for _ in 0..batch_size {
        for _ in 0..oh {
            for _ in 0..ow {
                for c in 0..c_out {
                    if inputs.len() == 3 {
                        biases_cells.push(biases[c].clone());
                    } else {
                        biases_cells.push(zero.clone());
                    }
                }
            }
        }
    }

    (input_cells, weights_cells, biases_cells)
}

pub fn splat_input<G: Clone + std::fmt::Debug>(
    input: &Array<G, IxDyn>,
    stride: (usize, usize),
    hh: usize,
    ww: usize,
    oh: usize,
    ow: usize,
) -> Vec<Vec<G>> {
    let (batch_size, c_in, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (si, sj) = stride;

    let mut splat = vec![];
    let mut row_idx = 0;
    for b in 0..batch_size {
        for k in 0..c_in {
            for i in 0..oh {
                for j in 0..ow {
                    splat.push(vec![]);
                    for ci in 0..hh {
                        for cj in 0..ww {
                            let ix = i * si + ci;
                            let jy = j * sj + cj;
                            if ix < h && jy < w {
                                splat[row_idx].push(input[[b, k, ix, jy]].clone());
                            }
                        }
                    }
                    row_idx += 1;
                }
            }
        }
    }
    splat
}

impl<F: PrimeField> ConvChip<F> {
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

        // input: batch * C_in * H * W
        let input = &inputs[0];
        // weight: C_out * C_in * K_h * K_w
        let weight = &inputs[1];

        let (batch_size, _c_in, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (c_out, hh, ww) = (weight.shape()[0], weight.shape()[2], weight.shape()[3]);
        let (si, sj) = stride;
        let (oh, ow) = (
            (h + ph.0 + ph.1 - hh) / si + 1,
            (w + pw.0 + pw.1 - ww) / sj + 1,
        );

        let (splat_inp, splat_weights, splat_biases) = splat(&inputs, 0 as Int, stride, ph, pw);

        let conv_size = splat_inp[0].len();
        let flattened_inp = splat_inp
            .into_iter()
            .flat_map(|x| x.into_iter())
            .collect::<Vec<_>>();
        let flattened_weights = splat_weights
            .into_iter()
            .flat_map(|x| x.into_iter())
            .collect::<Vec<_>>();

        let inp_array =
            Array::from_shape_vec(IxDyn(&vec![batch_size * oh * ow, conv_size]), flattened_inp)
                .unwrap();
        let weights_array =
            Array::from_shape_vec(IxDyn(&vec![c_out, conv_size]), flattened_weights).unwrap();
        let biases_array =
            Array::from_shape_vec(IxDyn(&vec![c_out, batch_size * oh * ow]), splat_biases).unwrap();

        let inp_array = inp_array.to_shape((batch_size * oh * ow, conv_size))?;
        let weights_array = weights_array.to_shape((c_out, conv_size))?;

        let outp_flat = weights_array.dot(&inp_array.t());
        let outp_flat = outp_flat.mapv(|x| fdiv(x, numeric_config.scale_factor as Int));
        let outp_flat = outp_flat + biases_array;

        let out_shape = vec![batch_size, c_out, oh, ow];
        let outp = outp_flat.to_shape(out_shape).unwrap();
        let outp = outp.to_owned();

        Ok(vec![outp])
    }

    // This function is used for non-circuit backward
    pub fn backward(
        inputs: &Vec<Tensor>,
        numeric_config: &NumericConfig,
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
        // batch * C_in * H * W
        let input = &inputs[1].clone();
        // C_out * C_in * K_h * K_w
        let weight = &inputs[2].clone();
        // batch * C_out * OH * OW
        let inpgrad = &inputs[0].clone();

        let (batch_size, c_in, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (c_out, hh, ww) = (weight.shape()[0], weight.shape()[2], weight.shape()[3]);
        let (oh, ow) = (inpgrad.shape()[2], inpgrad.shape()[3]);
        let (si, sj) = stride;
        let paddings = vec![[0, 0], [0, 0], [ph.0, ph.1], [pw.0, pw.1]];
        let inp_pad = pad(&input, paddings, &(0 as Int));

        let splat_input = splat_input(&inp_pad, stride, hh, ww, oh, ow);

        let flattened_inp = splat_input
            .into_iter()
            .flat_map(|x| x.into_iter())
            .collect::<Vec<_>>();
        let inp_array = Array::from_shape_vec(
            IxDyn(&vec![c_in * hh * ww, batch_size * oh * ow]),
            flattened_inp,
        )?;

        // db = np.sum(dout, axis=(0, 2, 3))
        let db = inpgrad
            .sum_axis(Axis(0))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1))
            .into_dyn();
        // println!("conv non-circuit backward db: {:?}", db.shape());

        // dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
        // dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)
        let dout_reshaped = inpgrad.to_shape((c_out, batch_size * oh * ow))?;
        let inp_array = inp_array.to_shape((c_in * hh * ww, batch_size * oh * ow))?;
        let dw = dout_reshaped.dot(&inp_array.t());
        let dw = dw.mapv(|x| fdiv(x, numeric_config.scale_factor as Int));
        let dw = dw.to_shape(weight.shape())?.into_owned();
        // println!("conv non-circuit backward dw: {:?}", dw.shape());

        // dx = weight_padding conv dout
        let w_reshaped = weight.to_shape((c_out, c_in * hh * ww))?;
        let dx_array = w_reshaped.t().dot(&dout_reshaped); // C_in * HH * WW x batch * oh * ow
        let dx_array = dx_array.mapv(|x| fdiv(x, numeric_config.scale_factor as Int));
        let dx_array = dx_array.to_shape((c_in, hh, ww, batch_size, oh, ow))?;

        let mut dx = Array4::<Int>::zeros((batch_size, c_in, h + ph.0 + ph.1, w + pw.0 + pw.1));

        for n in 0..batch_size {
            for c in 0..c_in {
                for hh in 0..hh {
                    for ww in 0..ww {
                        for h in 0..oh / si {
                            for w in 0..ow / sj {
                                let dx_idx = dx_array.slice(s![c, hh, ww, n, h, w]).sum();
                                dx[(n, c, si * h + hh, sj * w + ww)] += dx_idx;
                            }
                        }
                    }
                }
            }
        }
        if ph.0 > 0 || ph.1 > 0 || pw.0 > 0 || pw.1 > 0 {
            dx = dx
                .slice(s![.., .., ph.0..h + ph.0, pw.0..w + pw.0])
                .to_owned();
        }
        // println!("conv non-circuit backward dx: {:?}", dx.shape());

        Ok(vec![dx.into_dyn(), dw, db])
    }
}
impl<F: PrimeField> Assign<F> for ConvChip<F> {}
impl<F: PrimeField> Operation<F> for ConvChip<F> {
    fn forward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        random: &Vec<CellRc<F>>,
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

        let zero = constants.get(&0).unwrap();
        let input = inputs[0].clone();
        let weight = inputs[1].clone();

        let (batch_size, _c_in, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (c_out, hh, ww) = (weight.shape()[0], weight.shape()[2], weight.shape()[3]);
        let (si, sj) = stride;
        let (oh, ow) = (
            (h + ph.0 + ph.1 - hh) / si + 1,
            (w + pw.0 + pw.1 - ww) / sj + 1,
        );

        let inputs = inputs
            .iter()
            .map(|x| x.map(|x| Rc::clone(x)))
            .collect::<Vec<_>>();
        let (splat_inp, splat_weights, splat_biases) = splat(&inputs, zero.clone(), stride, ph, pw);

        let outp_flat: Vec<AssignedCell<F, F>> = {
            let fc_chip = GemmChip::<F>::construct(self.numeric_config.clone());

            let conv_size = splat_inp[0].len();
            let flattened_inp = splat_inp.into_iter().flat_map(|x| x.into_iter()).collect();
            let flattened_weights = splat_weights
                .into_iter()
                .flat_map(|x| x.into_iter())
                .collect::<Vec<_>>();

            let inp_array = Array::from_shape_vec(
                IxDyn(&vec![batch_size * oh * ow, conv_size]),
                flattened_inp,
            )?;
            let weights_array =
                Array::from_shape_vec(IxDyn(&vec![c_out, conv_size]), flattened_weights)?;
            let biases_array =
                Array::from_shape_vec(IxDyn(&vec![c_out, batch_size * oh * ow]), splat_biases)?;

            let outp_slice = fc_chip
                .forward(
                    layouter.namespace(|| ""),
                    &vec![
                        weights_array.view(),
                        inp_array.view().t(),
                        biases_array.view(),
                    ],
                    &constants,
                    &random,
                    &attributes,
                )
                .unwrap();

            let outp_flat = outp_slice[0]
                .clone()
                .into_iter()
                .map(|x| (*x).clone())
                .collect::<Vec<_>>();
            outp_flat
        };

        let out_shape = vec![batch_size, c_out, oh, ow];
        let outp_flat = outp_flat.iter().map(|x| Rc::new(x.clone())).collect();
        let outp = Array::from_shape_vec(IxDyn(&out_shape), outp_flat).unwrap();
        // println!("conv circuit forward output: {:?}", outp.shape());

        Ok(vec![outp.into_dyn()])
    }

    fn backward(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: &Vec<AssignedTensorRef<F>>,
        constants: &BTreeMap<Int, CellRc<F>>,
        random: &Vec<CellRc<F>>,
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

        let zero = constants.get(&0).unwrap();
        let inpgrad = inputs[0].clone();
        let input = inputs[1].clone();
        let weight = inputs[2].clone();

        let (batch_size, c_in, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (c_out, hh, ww) = (weight.shape()[0], weight.shape()[2], weight.shape()[3]);
        let (oh, ow) = (inpgrad.shape()[2], inpgrad.shape()[3]);
        let (si, sj) = stride;

        let paddings = vec![[0, 0], [0, 0], [ph.0, ph.1], [pw.0, pw.1]];
        let inp_pad = pad(&input.to_owned(), paddings, &zero);

        let mut splat_inp = vec![];
        let mut input_row_idx = 0;
        for b in 0..batch_size {
            for i in 0..oh {
                for j in 0..ow {
                    splat_inp.push(vec![]);
                    for ci in 0..hh {
                        for cj in 0..ww {
                            for ck in 0..c_in {
                                let idx_i = i * si + ci;
                                let idx_j = j * sj + cj;
                                splat_inp[input_row_idx]
                                    .push(inp_pad[[b, ck, idx_i, idx_j]].clone());
                            }
                        }
                    }
                    input_row_idx += 1;
                }
            }
        }

        let flattened_inp = splat_inp
            .into_iter()
            .flat_map(|x| x.into_iter())
            .collect::<Vec<_>>();
        let inp_array = Array::from_shape_vec(
            IxDyn(&vec![c_in * hh * ww, batch_size * oh * ow]),
            flattened_inp,
        )?;
        let inp_array = inp_array.view();

        let fc_chip = GemmChip::<F>::construct(self.numeric_config.clone());
        let sum = SumLayouter::construct(self.numeric_config.clone());
        let update = UpdateLayouter::construct(self.numeric_config.clone());

        // db
        // region
        let mut dout_cout = vec![];
        for c in 0..c_out {
            let mut row = vec![];
            for b in 0..batch_size {
                for oh in 0..oh {
                    for ow in 0..ow {
                        row.push(&inpgrad[[b, c, oh, ow]]);
                    }
                }
            }
            dout_cout.push(row);
        }
        // println!("conv circuit backward, dout_Cout shape: {:?} {:?}", dout_Cout.len(),dout_Cout[0].len());

        let db = layouter
            .assign_region(
                || "conv db sum",
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;
                    let mut output = vec![];

                    dout_cout.iter().for_each(|x| {
                        let x = x.clone().iter().map(|x| (*x).as_ref()).collect::<Vec<_>>();
                        let out = sum
                            .layout(region, row_offset, &vec![x], &vec![zero.as_ref()])
                            .unwrap();
                        row_offset = out.1;
                        output.extend(out.0);
                    });
                    Ok(output)
                },
            )
            .unwrap();
        let db = db.iter().map(|x| Rc::new(x.clone())).collect::<Vec<_>>();
        let db = Array::from_shape_vec(IxDyn(&vec![c_out]), db)?;
        // println!("conv circuit backward, db: {:?}", db.shape());
        // endregion

        // dw
        // region
        let dout = inpgrad.iter().map(|x| x.clone()).collect::<Vec<_>>();
        let dout_reshaped = Array::from_shape_vec(IxDyn(&vec![c_out, batch_size * oh * ow]), dout)?;
        let dout_reshaped = dout_reshaped.view();

        let splat_biases = vec![zero.clone(); dout_reshaped.shape()[0] * inp_array.t().shape()[1]];
        let biases_array = Array::from_shape_vec(
            IxDyn(&vec![dout_reshaped.shape()[0], inp_array.t().shape()[1]]),
            splat_biases,
        )
        .unwrap();
        let biases_array = biases_array.view();

        let dw_slice = fc_chip
            .forward(
                layouter.namespace(|| ""),
                &vec![dout_reshaped.clone(), inp_array.t(), biases_array.clone()],
                &constants,
                &random,
                &attributes,
            )
            .unwrap();

        let dw_flat = dw_slice[0]
            .clone()
            .into_iter()
            .map(|x| (*x).clone())
            .collect::<Vec<_>>();
        let dw_flat = dw_flat.iter().map(|x| Rc::new(x.clone())).collect();
        let dw = Array::from_shape_vec(IxDyn(weight.shape()), dw_flat)?;
        // println!("conv circuit backward, dw: {:?}", dw.shape());

        // Update weight
        let new_dw = layouter
            .assign_region(
                || "conv update weight",
                |mut region| {
                    let region = &mut region;
                    let row_offset = 0;

                    let new_weight = match update.layout(
                        region,
                        row_offset,
                        &vec![
                            weight.iter().map(|x| x.as_ref()).collect(),
                            dw.iter().map(|x| x.as_ref()).collect(),
                        ],
                        &vec![zero.as_ref()],
                    ) {
                        Ok(output) => {
                            // row_offset = output.1;
                            output.0
                        }
                        Err(e) => panic!("Error in convChip.update: {:?}", e),
                    };
                    Ok(new_weight)
                },
            )
            .unwrap();
        let dw = Array::from_shape_vec(
            IxDyn(&weight.shape()),
            new_dw.into_iter().map(|x| Rc::new(x)).collect(),
        )?;
        // println!("conv circuit backward, after update weight, dw: {:?}", dw.shape());
        // endregion

        // dx
        // region
        let w_reshaped = weight.iter().map(|x| x.clone()).collect::<Vec<_>>();
        let w_reshaped = Array::from_shape_vec(IxDyn(&vec![c_out, c_in * hh * ww]), w_reshaped)?;
        let w_reshaped = w_reshaped.view();
        let splat_biases = vec![zero.clone(); w_reshaped.t().shape()[0] * dout_reshaped.shape()[1]];
        let biases_array = Array::from_shape_vec(
            IxDyn(&vec![w_reshaped.t().shape()[0], dout_reshaped.shape()[1]]),
            splat_biases,
        )
        .unwrap();
        let biases_array = biases_array.view();

        let dx_slice = fc_chip
            .forward(
                layouter.namespace(|| ""),
                &vec![w_reshaped.t(), dout_reshaped, biases_array],
                &constants,
                &random,
                &attributes,
            )
            .unwrap();
        let dx_flat = dx_slice[0]
            .clone()
            .into_iter()
            .map(|x| (*x).clone())
            .collect::<Vec<_>>();
        let dx_flat = dx_flat.iter().map(|x| Rc::new(x.clone())).collect();
        let dx_array =
            Array::from_shape_vec(IxDyn(&vec![c_in, hh, ww, batch_size, oh, ow]), dx_flat)?;

        let add = AddLayouter::construct(self.numeric_config.clone());

        let dx_zeros_vec = vec![0; batch_size * c_in * (h + ph.0 + ph.1) * (w + pw.0 + pw.1)];
        let f_dx_zeros = dx_zeros_vec
            .iter()
            .map(|x| to_field::<F>(*x))
            .collect::<Vec<_>>();
        let f_dx_zeros = Array::from_shape_vec(
            (batch_size, c_in, h + ph.0 + ph.1, w + pw.0 + pw.1),
            f_dx_zeros,
        )?
        .into_dyn();
        let dx_zeros = self
            .assign_tensor(
                layouter.namespace(|| "assign_dx_zeros"),
                &self.numeric_config.columns,
                &f_dx_zeros,
            )
            .unwrap();

        let mut dx = layouter
            .assign_region(
                || "conv dx sum",
                |mut region| {
                    let region = &mut region;
                    let mut row_offset = 0;

                    let dx = dx_zeros
                        .iter()
                        .map(|x| x.as_ref().clone())
                        .collect::<Vec<_>>();
                    let mut dx = Array::from_shape_vec(
                        (batch_size, c_in, h + ph.0 + ph.1, w + pw.0 + pw.1),
                        dx,
                    )
                    .unwrap()
                    .into_dyn();

                    for n in 0..batch_size {
                        for c in 0..c_in {
                            for hh in 0..hh {
                                for ww in 0..ww {
                                    for h in 0..oh / si {
                                        for w in 0..ow / sj {
                                            // println!("  {:?} {:?} {:?} {:?}", n, c, (si * h + hh), (sj * w + ww));
                                            let inp1 = dx_array.slice(s![c, hh, ww, n, h, w]);
                                            let inp1 =
                                                inp1.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                                            let inp2 = vec![&dx[[n, c, si * h + hh, sj * w + ww]]];

                                            let add_res = add
                                                .layout(
                                                    region,
                                                    row_offset,
                                                    &vec![inp1, inp2],
                                                    &vec![zero.as_ref()],
                                                )
                                                .unwrap();
                                            row_offset = add_res.1;
                                            dx[[n, c, si * h + hh, sj * w + ww]] =
                                                add_res.0[0].clone();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(dx)
                },
            )
            .unwrap();

        if ph.0 > 0 || ph.1 > 0 || pw.0 > 0 || pw.1 > 0 {
            let tmp = dx
                .slice(s![.., .., ph.0..(h + ph.0), pw.0..(w + pw.0)])
                .iter()
                .map(|x| x.clone())
                .collect::<Vec<_>>();
            dx = Array::from_shape_vec(input.shape(), tmp)?.into_dyn();
        }

        let dx = dx.iter().map(|x| Rc::new(x.clone())).collect::<Vec<_>>();
        let dx = Array::from_shape_vec(IxDyn(&input.shape()), dx)?;
        // println!("conv circuit backward, dx: {:?}", dx.shape());
        // endregion

        Ok(vec![dx, dw, db])
    }
}

impl<F: PrimeField> NumericConsumer for ConvChip<F> {
    fn used_numerics(&self) -> Vec<NumericType> {
        vec![
            NumericType::Sum,
            NumericType::Add,
            NumericType::Update,
            NumericType::DotVec,
            NumericType::DivSF,
        ]
    }
}
