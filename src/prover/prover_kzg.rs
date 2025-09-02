use std::{fs::File, io::Write, time::Instant};

use halo2_proofs::{
    halo2curves::{
        bn256::{Bn256, Fr, G1Affine},
        ff::{FromUniformBytes, PrimeField},
    },
    plonk::{create_wrapped_proof, keygen_pk, keygen_vk, verify_proof, ProvingKey, VerifyingKey},
    poly::{
        commitment::Params,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use rand::rngs::OsRng;

use crate::stage::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit};

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
        println!("Constructing KZGProver");
        let fparams = Self::create_params(k);
        // let gparams = ParamsKZG::setup(k as u32, rng.clone());
        // let bparams = ParamsKZG::setup(k as u32, rng.clone());
        let gparams = fparams.clone();
        let bparams = fparams.clone();

        Self {
            fparams,
            gparams,
            bparams,
            stage: Default::default(),
        }
    }

    pub fn construct_with_params(
        fparams: ParamsKZG<Bn256>,
        gparams: ParamsKZG<Bn256>,
        bparams: ParamsKZG<Bn256>,
    ) -> Self {
        println!("Constructing KZGProver with params");
        Self {
            fparams,
            gparams,
            bparams,
            stage: Default::default(),
        }
    }

    pub fn create_params(k: usize) -> ParamsKZG<Bn256> {
        println!("Creating ParamsKZG");
        let start = Instant::now();
        let rng = rand::thread_rng();
        let params = ParamsKZG::<Bn256>::setup(k as u32, rng);
        println!("ParamsKZG::setup took {:?}", start.elapsed());

        let mut buf = Vec::new();
        params.write(&mut buf).expect("Failed to write params");
        let params_path = format!("./params/kzg_{}.params", k);
        let mut file = File::create(&params_path).expect("Failed to create params file");
        file.write_all(&buf[..])
            .expect("Failed to write params to file");
        params
    }

    pub fn load_params(params_path: String) -> ParamsKZG<Bn256> {
        println!("Loading ParamsKZG");
        let start = Instant::now();
        let mut params_fs = File::open(&params_path).expect("couldn't load params");
        let params = ParamsKZG::read(&mut params_fs).expect("Failed to read params");
        println!("ParamsKZG::read took {:?}", start.elapsed());
        params
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
        let mut rng = OsRng;
        let start = Instant::now();
        println!("Proving");
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        match stage {
            StageType::Forward => {
                let commitments =
                    create_wrapped_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
                        &self.fparams,
                        &pk,
                        &[self.stage.forward.as_ref().unwrap().clone()],
                        &[&[&public]],
                        &mut rng,
                        &mut transcript,
                        1,
                        vec![(0, 10), (10, 20)],
                    )
                    .expect("proof generation should not fail");

                println!("commitments: {:?}", commitments);
            }
            StageType::Gradient => {
                create_wrapped_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
                    &self.gparams,
                    &pk,
                    &[self.stage.gradient.as_ref().unwrap().clone()],
                    &[&[&public]],
                    &mut rng,
                    &mut transcript,
                    1,
                    vec![],
                )
                .expect("proof generation should not fail");
            }
            StageType::Backward => {
                create_wrapped_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
                    &self.bparams,
                    &pk,
                    &[self.stage.backward.as_ref().unwrap().clone()],
                    &[&[&public]],
                    &mut rng,
                    &mut transcript,
                    1,
                    vec![],
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
        let res = verify_proof::<_, VerifierSHPLONK<_>, _, _, _>(
            &params,
            vk,
            strategy,
            &[&[&public]],
            &mut transcript,
            params.n(),
        )
        .is_ok();
        println!("Verifying took {:?}", start.elapsed());

        res
    }
}
