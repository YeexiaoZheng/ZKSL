use std::time::Instant;

use halo2_proofs::{
    halo2curves::{
        bn256::{Bn256, Fr, G1Affine},
        ff::{FromUniformBytes, PrimeField},
    },
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

use crate::stages::{
    backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit,
};

pub enum StageType {
    Forward,
    Gradient,
    Backward,
}

#[derive(Debug, Clone, Default)]
pub struct Stage<F: PrimeField + Ord + FromUniformBytes<64>> {
    pub forward: Option<ForwardCircuit<F>>,
    pub gradient: Option<GradientCircuit<F>>,
    pub backward: Option<BackwardCircuit<F>>,
}

pub struct KZGProver {
    pub stage: Stage<Fr>,
    pub fparams: ParamsKZG<Bn256>,
    pub gparams: ParamsKZG<Bn256>,
    pub bparams: ParamsKZG<Bn256>,
}

impl KZGProver {
    pub fn construct(k: usize) -> Self {
        let rng = rand::thread_rng();
        println!("Constructing KZGProver");
        let start = Instant::now();
        let fparams = ParamsKZG::setup(k as u32, rng.clone());
        let gparams = ParamsKZG::setup(k as u32, rng.clone());
        let bparams = ParamsKZG::setup(k as u32, rng.clone());
        println!("ParamsKZG::setup took {:?}", start.elapsed());

        Self {
            fparams,
            gparams,
            bparams,
            stage: Default::default(),
        }
    }

    pub fn set_forward(&mut self, forward: ForwardCircuit<Fr>) {
        self.stage.forward = Some(forward);
    }

    pub fn set_gradient(&mut self, gradient: GradientCircuit<Fr>) {
        self.stage.gradient = Some(gradient);
    }

    pub fn set_backward(&mut self, backward: BackwardCircuit<Fr>) {
        self.stage.backward = Some(backward);
    }

    pub fn keygen_forward(&self) -> (ProvingKey<G1Affine>, VerifyingKey<G1Affine>) {
        println!("Generating PK and VK");
        let start = Instant::now();
        let vk = keygen_vk(&self.fparams, self.stage.forward.as_ref().unwrap())
            .expect("keygen_vk should not fail");
        println!("keygen_vk took {:?}", start.elapsed());
        let pk = keygen_pk(
            &self.fparams,
            vk.clone(),
            self.stage.forward.as_ref().unwrap(),
        )
        .expect("keygen_pk should not fail");
        println!("keygen_pkvk took {:?}", start.elapsed());
        (pk, vk)
    }

    pub fn keygen_gradient(&self) -> (ProvingKey<G1Affine>, VerifyingKey<G1Affine>) {
        println!("Generating PK and VK");
        let start = Instant::now();
        let vk = keygen_vk(&self.gparams, self.stage.gradient.as_ref().unwrap())
            .expect("keygen_vk should not fail");
        println!("keygen_vk took {:?}", start.elapsed());
        let pk = keygen_pk(
            &self.gparams,
            vk.clone(),
            self.stage.gradient.as_ref().unwrap(),
        )
        .expect("keygen_pk should not fail");
        println!("keygen_pkvk took {:?}", start.elapsed());
        (pk, vk)
    }

    pub fn keygen_backward(&self) -> (ProvingKey<G1Affine>, VerifyingKey<G1Affine>) {
        println!("Generating PK and VK");
        let start = Instant::now();
        let vk = keygen_vk(&self.bparams, self.stage.backward.as_ref().unwrap())
            .expect("keygen_vk should not fail");
        println!("keygen_vk took {:?}", start.elapsed());
        let pk = keygen_pk(
            &self.bparams,
            vk.clone(),
            self.stage.backward.as_ref().unwrap(),
        )
        .expect("keygen_pk should not fail");
        println!("keygen_pkvk took {:?}", start.elapsed());
        (pk, vk)
    }

    pub fn prove(&self, stage: StageType, pk: &ProvingKey<G1Affine>, public: Vec<Fr>) -> Vec<u8> {
        let rng = rand::thread_rng();
        let start = Instant::now();
        println!("Proving");
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        match stage {
            StageType::Forward => {
                create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
                    &self.fparams,
                    &pk,
                    &[self.stage.forward.as_ref().unwrap().clone()],
                    &[&[&public]],
                    rng,
                    &mut transcript,
                )
                .expect("proof generation should not fail");
            }
            StageType::Gradient => {
                create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
                    &self.gparams,
                    &pk,
                    &[self.stage.gradient.as_ref().unwrap().clone()],
                    &[&[&public]],
                    rng,
                    &mut transcript,
                )
                .expect("proof generation should not fail");
            }
            StageType::Backward => {
                create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
                    &self.bparams,
                    &pk,
                    &[self.stage.backward.as_ref().unwrap().clone()],
                    &[&[&public]],
                    rng,
                    &mut transcript,
                )
                .expect("proof generation should not fail");
            }
        }
        let proof: Vec<u8> = transcript.finalize();
        println!("Proving took {:?}", start.elapsed());

        proof
    }

    pub fn verify(
        &self,
        stage: StageType,
        vk: &VerifyingKey<G1Affine>,
        public: Vec<Fr>,
        proof: Vec<u8>,
    ) -> bool {
        let start = Instant::now();
        println!("Verifying");
        let params = match stage {
            StageType::Forward => &self.fparams,
            StageType::Gradient => &self.gparams,
            StageType::Backward => &self.bparams,
        };
        let strategy = SingleStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let res = verify_proof::<KZGCommitmentScheme<_>, VerifierSHPLONK<_>, _, _, _>(
            &params,
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
