use std::marker::PhantomData;

use halo2_poseidon::poseidon::{
    primitives::{generate_constants, ConstantLength, Mds, Spec},
    Hash, Pow5Chip, Pow5Config,
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    halo2curves::ff::{FromUniformBytes, PrimeField},
    plonk::{Advice, Column, ConstraintSystem, Error},
};

use crate::numerics::numeric::NumericConfig;

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

#[derive(Debug, Clone)]
pub struct PoseidonChip<F, S, const WIDTH: usize, const RATE: usize>
where
    F: PrimeField + Ord + FromUniformBytes<64>,
    S: Spec<F, WIDTH, RATE> + Copy + Clone,
{
    poseidon_config: Pow5Config<F, WIDTH, RATE>,
    _marker: PhantomData<F>,
    _spec: PhantomData<S>,
}

impl<F, S, const WIDTH: usize, const RATE: usize> PoseidonChip<F, S, WIDTH, RATE>
where
    F: PrimeField + Ord + FromUniformBytes<64>,
    S: Spec<F, WIDTH, RATE> + Copy + Clone,
{
    pub fn configure(
        _numeric_config: NumericConfig,
        meta: &mut ConstraintSystem<F>,
        _input: [Column<Advice>; RATE],
        state: [Column<Advice>; WIDTH],
        partial_sbox: Column<Advice>,
    ) -> Option<Self> {
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
            _marker: PhantomData,
            _spec: PhantomData,
        })
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
