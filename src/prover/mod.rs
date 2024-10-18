pub mod prover_kzg;


// prover types and traits are defined here
// use std::marker::PhantomData;

// use halo2_proofs::{
//     halo2curves::{
//         ff::{FromUniformBytes, PrimeField},
//         pasta::EqAffine,
//     },
//     plonk::{keygen_pk, keygen_vk, Circuit},
//     poly::{commitment::Params, kzg::commitment::ParamsKZG},
//     transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
// };

// use crate::stages::forward::ForwardCircuit;

// pub struct Prover<F: PrimeField + Ord + FromUniformBytes<64>> {
//     _marker: PhantomData<F>,
// }

// pub fn gen_pk_vk<F: PrimeField + Ord + FromUniformBytes<64>>(circuit: &ForwardCircuit<F>) {
//     let params: Params<EqAffine> = Params::new(3);
//     let vk = keygen_vk(&params, circuit).expect("keygen_vk should not fail");
//     let pk = keygen_pk(&params, vk, circuit).expect("keygen_pk should not fail");
//     let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
// }
