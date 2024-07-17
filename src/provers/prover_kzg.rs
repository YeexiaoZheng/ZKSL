use std::time::Instant;

use halo2_proofs::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, ProvingKey, VerifyingKey},
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use rand::rngs::OsRng;

use crate::stages::forward::ForwardCircuit;

pub struct KZGProver {
    pub forward_circuit: ForwardCircuit<Fr>,
    pub params: ParamsKZG<Bn256>,
}

impl KZGProver {
    pub fn construct(forward_circuit: ForwardCircuit<Fr>) -> Self {
        println!("Constructing KZGProver");
        let start = Instant::now();
        let params = ParamsKZG::setup(forward_circuit.k as u32, OsRng);
        println!("ParamsKZG::setup took {:?}", start.elapsed());

        Self {
            params,
            forward_circuit,
        }
    }

    pub fn gen_pk_vk(&self) -> (ProvingKey<G1Affine>, VerifyingKey<G1Affine>) {
        println!("Generating PK and VK");
        let start = Instant::now();
        let vk = keygen_vk(&self.params, &self.forward_circuit).expect("keygen_vk should not fail");
        println!("keygen_vk took {:?}", start.elapsed());
        let pk = keygen_pk(&self.params, vk.clone(), &self.forward_circuit)
            .expect("keygen_pk should not fail");
        println!("keygen_pkvk took {:?}", start.elapsed());
        (pk, vk)
    }

    pub fn prove(&self, pk: &ProvingKey<G1Affine>, public: Vec<Fr>) -> Vec<u8> {
        let rng = rand::thread_rng();
        let start = Instant::now();
        println!("Proving");
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
            &self.params,
            &pk,
            &[self.forward_circuit.clone()],
            &[&[&public]],
            rng,
            &mut transcript,
        )
        .expect("proof generation should not fail");
        let proof: Vec<u8> = transcript.finalize();
        println!("Proving took {:?}", start.elapsed());

        proof
    }

    pub fn verify(&self, vk: &VerifyingKey<G1Affine>, public: Vec<Fr>, proof: Vec<u8>) -> bool {
        let start = Instant::now();
        println!("Verifying");
        let strategy = SingleStrategy::new(&self.params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let res = verify_proof::<KZGCommitmentScheme<_>, VerifierSHPLONK<_>, _, _, _>(
            &self.params,
            vk,
            strategy,
            &[&[&public]],
            &mut transcript,
        )
        .is_ok();
        println!("Verifying took {:?}", start.elapsed());

        res
    }
}
