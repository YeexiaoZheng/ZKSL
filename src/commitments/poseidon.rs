use std::{marker::PhantomData, rc::Rc};

use halo2_poseidon::poseidon::{
    self,
    primitives::{generate_constants, ConstantLength, Mds, Spec},
    Hash, Pow5Chip, Pow5Config,
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    halo2curves::ff::{FromUniformBytes, PrimeField},
    plonk::{Advice, Column, ConstraintSystem, Error},
};

use crate::{numeric::NumericConfig, utils::helpers::to_field};

pub const WIDTH: usize = 6;
pub const RATE: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct PoseidonSpec<F: PrimeField, const WIDTH: usize, const RATE: usize> {
    _marker: PhantomData<F>,
}

impl<F, const WIDTH: usize, const RATE: usize> Spec<F, WIDTH, RATE> for PoseidonSpec<F, WIDTH, RATE>
where
    F: PrimeField + Ord + FromUniformBytes<64>,
{
    fn full_rounds() -> usize {
        8
    }

    fn partial_rounds() -> usize {
        56
    }

    fn sbox(val: F) -> F {
        val.pow_vartime([5])
    }

    fn secure_mds() -> usize {
        0
    }

    fn constants() -> (Vec<[F; WIDTH]>, Mds<F, WIDTH>, Mds<F, WIDTH>) {
        generate_constants::<_, Self, WIDTH, RATE>()
    }
}

pub struct PoseidonHash<F: PrimeField + Ord + FromUniformBytes<64>>(PhantomData<F>);
impl<F> PoseidonHash<F>
where
    F: PrimeField + Ord + FromUniformBytes<64>,
{
    pub fn hash_vec_to_one(inputs: Vec<F>) -> F {
        let hasher = poseidon::primitives::Hash::<
            _,
            PoseidonSpec<F, WIDTH, RATE>,
            ConstantLength<RATE>,
            WIDTH,
            RATE,
        >::init();
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(to_field::<F>(0));
        }
        let mut hash_output = vec![];
        for chunk in inputs.chunks(RATE) {
            hash_output.push(hasher.clone().hash(chunk.try_into().unwrap()));
        }
        while hash_output.len() > 1 {
            let mut new_output = vec![];
            while hash_output.len() % RATE != 0 {
                hash_output.push(to_field::<F>(0));
            }
            for chunk in hash_output.chunks(RATE) {
                new_output.push(hasher.clone().hash(chunk.try_into().unwrap()));
            }
            hash_output = new_output;
        }
        hash_output[0]
    }

    pub fn hash_vec(inputs: Vec<F>) -> Vec<F> {
        let hasher = poseidon::primitives::Hash::<
            _,
            PoseidonSpec<F, WIDTH, RATE>,
            ConstantLength<RATE>,
            WIDTH,
            RATE,
        >::init();
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(to_field::<F>(0));
        }
        inputs
            .chunks(RATE)
            .map(|chunk| hasher.clone().hash(chunk.try_into().unwrap()))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct PoseidonChip<F, S, const WIDTH: usize, const RATE: usize>
where
    F: PrimeField + Ord + FromUniformBytes<64>,
    S: Spec<F, WIDTH, RATE> + Copy + Clone,
{
    poseidon_config: Pow5Config<F, WIDTH, RATE>,
    _spec: PhantomData<S>,
}

impl<F, S, const WIDTH: usize, const RATE: usize> PoseidonChip<F, S, WIDTH, RATE>
where
    F: PrimeField + Ord + FromUniformBytes<64>,
    S: Spec<F, WIDTH, RATE> + Copy + Clone,
{
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> Option<Self> {
        let state: [Column<Advice>; WIDTH] = numeric_config.columns[RATE..RATE + WIDTH]
            .try_into()
            .unwrap();
        let partial_sbox = numeric_config.columns[RATE + WIDTH].try_into().unwrap();

        let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

        meta.enable_constant(rc_b[0]);

        Some(Self {
            poseidon_config: Pow5Chip::configure::<S>(
                meta,
                state.try_into().unwrap(),
                partial_sbox,
                rc_a.try_into().unwrap(),
                rc_b.try_into().unwrap(),
            ),
            _spec: PhantomData,
        })
    }

    pub fn hash_vec_non_circuit(inputs: Vec<F>) -> Vec<F> {
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(to_field::<F>(0));
        }
        let mut hash_output = vec![];
        for chunk in inputs.chunks(RATE) {
            hash_output.push(
                poseidon::primitives::Hash::<
                    _,
                    PoseidonSpec<F, WIDTH, RATE>,
                    ConstantLength<RATE>,
                    WIDTH,
                    RATE,
                >::init()
                .hash(chunk.try_into().unwrap()),
            )
        }
        hash_output
    }

    pub fn hash_vec(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: Vec<Rc<AssignedCell<F, F>>>,
        zero: Rc<AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(zero.clone());
        }
        let chip = Pow5Chip::construct(self.poseidon_config.clone());
        let hasher = Hash::<_, _, S, ConstantLength<RATE>, WIDTH, RATE>::init(
            chip.clone(),
            layouter.namespace(|| "init hasher"),
        )?;

        let mut output = vec![];
        for (i, input) in inputs.chunks(RATE).enumerate() {
            output.push(
                hasher.clone().hash(
                    layouter.namespace(|| format!("{}", i)),
                    input
                        .iter()
                        .map(|x| x.as_ref().clone())
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                )?,
            );
        }

        Ok(output)
    }

    pub fn hash(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: [AssignedCell<F, F>; RATE],
    ) -> Result<AssignedCell<F, F>, Error> {
        let chip = Pow5Chip::construct(self.poseidon_config.clone());
        let hasher = Hash::<_, _, S, ConstantLength<RATE>, WIDTH, RATE>::init(
            chip,
            layouter.namespace(|| "init hasher"),
        )
        .unwrap();
        hasher.hash(layouter.namespace(|| "poseidon hash"), inputs)
    }
}

#[derive(Debug, Clone)]
pub struct FixedPoseidonChip<F>
where
    F: PrimeField + Ord + FromUniformBytes<64>,
{
    poseidon_config: Pow5Config<F, WIDTH, RATE>,
    _spec: PhantomData<F>,
}

impl<F> FixedPoseidonChip<F>
where
    F: PrimeField + Ord + FromUniformBytes<64>,
{
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        numeric_config: NumericConfig,
    ) -> Option<Self> {
        if numeric_config.commitment {
            let state: [Column<Advice>; WIDTH] = numeric_config.columns[RATE..RATE + WIDTH]
                .try_into()
                .unwrap();
            let partial_sbox = numeric_config.columns[RATE + WIDTH].try_into().unwrap();

            let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
            let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

            meta.enable_constant(rc_b[0]);

            Some(Self {
                poseidon_config: Pow5Chip::configure::<PoseidonSpec<F, WIDTH, RATE>>(
                    meta,
                    state.try_into().unwrap(),
                    partial_sbox,
                    rc_a.try_into().unwrap(),
                    rc_b.try_into().unwrap(),
                ),
                _spec: PhantomData,
            })
        } else {
            None
        }
    }

    pub fn hash_vec_to_one_non_circuit(inputs: Vec<F>) -> F {
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(to_field::<F>(0));
        }
        let mut hash_output = vec![];
        for chunk in inputs.chunks(RATE) {
            hash_output.push(
                poseidon::primitives::Hash::<
                    _,
                    PoseidonSpec<F, WIDTH, RATE>,
                    ConstantLength<RATE>,
                    WIDTH,
                    RATE,
                >::init()
                .hash(chunk.try_into().unwrap()),
            )
        }
        while hash_output.len() > 1 {
            let mut new_output = vec![];
            while hash_output.len() % RATE != 0 {
                hash_output.push(to_field::<F>(0));
            }
            for chunk in hash_output.chunks(RATE) {
                new_output.push(
                    poseidon::primitives::Hash::<
                        _,
                        PoseidonSpec<F, WIDTH, RATE>,
                        ConstantLength<RATE>,
                        WIDTH,
                        RATE,
                    >::init()
                    .hash(chunk.try_into().unwrap()),
                )
            }
            hash_output = new_output;
        }
        hash_output[0]
    }

    pub fn hash_vec_non_circuit(inputs: Vec<F>) -> Vec<F> {
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(to_field::<F>(0));
        }
        let mut hash_output = vec![];
        for chunk in inputs.chunks(RATE) {
            hash_output.push(
                poseidon::primitives::Hash::<
                    _,
                    PoseidonSpec<F, WIDTH, RATE>,
                    ConstantLength<RATE>,
                    WIDTH,
                    RATE,
                >::init()
                .hash(chunk.try_into().unwrap()),
            )
        }
        hash_output
    }

    pub fn hash_vec_to_one(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: Vec<Rc<AssignedCell<F, F>>>,
        zero: Rc<AssignedCell<F, F>>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(zero.clone());
        }

        let chip = Pow5Chip::construct(self.poseidon_config.clone());
        let hasher =
            Hash::<_, _, PoseidonSpec<F, WIDTH, RATE>, ConstantLength<RATE>, WIDTH, RATE>::init(
                chip.clone(),
                layouter.namespace(|| "init hasher"),
            )?;

        let mut output = vec![];
        for (i, input) in inputs.chunks(RATE).enumerate() {
            output.push(
                hasher.clone().hash(
                    layouter.namespace(|| format!("{}", i)),
                    input
                        .iter()
                        .map(|x| x.as_ref().clone())
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                )?,
            );
        }

        while output.len() > 1 {
            let mut new_output = vec![];
            while output.len() % RATE != 0 {
                output.push(zero.as_ref().clone());
            }
            for (i, input) in output.chunks(RATE).enumerate() {
                new_output.push(
                    hasher.clone().hash(
                        layouter.namespace(|| format!("{}", i)),
                        input
                            .iter()
                            .map(|x| x.clone())
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap(),
                    )?,
                );
            }
            output = new_output;
        }

        Ok(output[0].clone())
    }

    pub fn hash_vec(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: Vec<Rc<AssignedCell<F, F>>>,
        zero: Rc<AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(zero.clone());
        }
        let chip = Pow5Chip::construct(self.poseidon_config.clone());
        let hasher =
            Hash::<_, _, PoseidonSpec<F, WIDTH, RATE>, ConstantLength<RATE>, WIDTH, RATE>::init(
                chip.clone(),
                layouter.namespace(|| "init hasher"),
            )?;

        let mut output = vec![];
        for (i, input) in inputs.chunks(RATE).enumerate() {
            output.push(
                hasher.clone().hash(
                    layouter.namespace(|| format!("{}", i)),
                    input
                        .iter()
                        .map(|x| x.as_ref().clone())
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                )?,
            );
        }

        Ok(output)
    }

    pub fn hash(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: [AssignedCell<F, F>; RATE],
    ) -> Result<AssignedCell<F, F>, Error> {
        let chip = Pow5Chip::construct(self.poseidon_config.clone());
        let hasher =
            Hash::<_, _, PoseidonSpec<F, WIDTH, RATE>, ConstantLength<RATE>, WIDTH, RATE>::init(
                chip,
                layouter.namespace(|| "init hasher"),
            )
            .unwrap();
        hasher.hash(layouter.namespace(|| "poseidon hash"), inputs)
    }
}

// #[derive(Debug, Clone)]
// pub struct FixedPoseidonConfig<F>
// where
//     F: PrimeField + Ord + FromUniformBytes<64>,
// {
//     pub numeric_config: NumericConfig,
//     pub hasher: FixedPoseidonChip<F>,
//     pub _marker: PhantomData<F>,
// }

// pub struct FixedPoseidonCircuit<F>
// where
//     F: PrimeField + Ord + FromUniformBytes<64>,
// {
//     pub input: Vec<F>,
//     pub output: Vec<F>,
// }

// impl<F> FixedPoseidonCircuit<F>
// where
//     F: PrimeField + Ord + FromUniformBytes<64>,
// {
//     pub fn construct(input: Vec<F>) -> Self {
//         Self {
//             input,
//             output: vec![],
//         }
//     }
//     pub fn modify_output(&mut self, output: Vec<F>) {
//         self.output = output;
//     }
// }

// impl<F> Circuit<F> for FixedPoseidonCircuit<F>
// where
//     F: PrimeField + Ord + FromUniformBytes<64>,
// {
//     type Config = FixedPoseidonConfig<F>;
//     type FloorPlanner = SimpleFloorPlanner;

//     fn without_witnesses(&self) -> Self {
//         todo!()
//     }

//     fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
//         let columns = (0..WIDTH + RATE + 1)
//             .map(|_| meta.advice_column())
//             .collect::<Vec<_>>();
//         let constants = vec![meta.fixed_column()];
//         for col in columns.iter() {
//             meta.enable_equality(*col);
//         }
//         for cst in constants.iter() {
//             meta.enable_equality(*cst);
//         }
//         let numeric_config = NumericConfig {
//             columns,
//             constants,
//             ..NumericConfig::default()
//         };

//         Self::Config {
//             numeric_config: numeric_config.clone(),
//             hasher: FixedPoseidonChip::configure(meta, numeric_config).unwrap(),
//             _marker: PhantomData,
//         }
//     }

//     fn synthesize(
//         &self,
//         config: Self::Config,
//         mut layouter: impl Layouter<F>,
//     ) -> Result<(), Error> {
//         let columns = config.numeric_config.columns;
//         let input = layouter.assign_region(
//             || "assign input",
//             |mut region| {
//                 let mut cell_idx = 0;
//                 self.input
//                     .iter()
//                     .map(|cell| {
//                         let row_idx = cell_idx / columns.len();
//                         let col_idx = cell_idx % columns.len();
//                         cell_idx += 1;
//                         let out = region.assign_advice(
//                             || "assign tensor cell",
//                             columns[col_idx],
//                             row_idx,
//                             || Value::known(*cell),
//                         )?;
//                         Ok(Rc::new(out))
//                     })
//                     .collect::<Result<Vec<_>, Error>>()
//             },
//         )?;

//         let zero = layouter.assign_region(
//             || "constants",
//             |mut region| {
//                 region.assign_fixed(
//                     || "zero",
//                     config.numeric_config.constants[0],
//                     0,
//                     || Value::known(F::ZERO),
//                 )
//             },
//         )?;
//         let res = config.hasher.hash_vec(layouter, input, Rc::new(zero))?;

//         let mut output = vec![];
//         let _ = res.iter().map(|o| {
//             o.value().map(|x| {
//                 output.push(x.clone());
//             });
//         });

//         // self.modify_output(output);

//         Ok(())
//     }
// }
