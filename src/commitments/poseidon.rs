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

use crate::{numerics::numeric::NumericConfig, utils::helpers::to_field};

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
    pub fn hash_vec(inputs: Vec<F>) -> Vec<F> {
        let mut inputs = inputs.clone();
        while inputs.len() % RATE != 0 {
            inputs.push(to_field::<F>(0));
        }
        inputs
            .chunks(RATE)
            .map(|chunk| {
                poseidon::primitives::Hash::<
                    _,
                    PoseidonSpec<F, WIDTH, RATE>,
                    ConstantLength<RATE>,
                    WIDTH,
                    RATE,
                >::init()
                .hash(chunk.try_into().unwrap())
            })
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
